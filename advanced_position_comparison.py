"""Paired, same-provider position experiments for appended notebook sections."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from enrich_player_biometrics import normalized

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'advanced_position_results'
CLASSES = ['DF','MF','FW']
BASE_MAPPING = {
    'Sh': ('Shots',['shotsTotal']), 'SoT': ('Shots',['shotOnTarget']),
    'Gls': ('Goals',['goalTotal']), 'Ast': ('Assists',['assist']),
    'Crs': ('Passes',['passCrossAccurate','passCrossInaccurate']),
    'Int': ('Interception',['interceptionAll']), 'TklW': ('Tackles',['tackleWonTotal']),
    'foulGiven': ('Fouls',['foulGiven']), 'foulCommitted': ('Fouls',['foulCommitted']),
    'Off': ('Offsides',['offsideGiven']), 'CrdY': ('Cards',['yellowCard']),
}
BASE_FEATURES = [name+'_p90' for name in BASE_MAPPING] + ['Age']
ADVANCED_FEATURES = [
    'key_passes_p90','passes_p90','accurate_passes_p90','pass_completion_pct','long_balls_p90',
    'successful_dribbles_p90','aerial_duels_won_p90','clearances_p90','blocks_p90',
    'dispossessions_p90','poor_controls_p90',
]
CLUB_ALIASES = {
    'Coventry':'Coventry City', 'Hull':'Hull City', 'Ipswich':'Ipswich Town', 'Leeds':'Leeds United',
    'Manchester United':'Manchester Utd', 'Nottingham Forest':'Nottingham',
    'Deportivo Alaves':'Alavés', 'Atletico Madrid':'Atlético Madrid',
    'Deportivo de A Coruna':'Dep. A Coruña', 'Malaga':'Málaga', 'Racing Santander':'Racing Sant',
    'AC Milan':'Milan', 'Parma Calcio 1913':'Parma', 'Paris Saint-Germain':'Paris SG',
    'Borussia Dortmund':'Dortmund', 'Eintracht Frankfurt':'Frankfurt',
    'Borussia M.Gladbach':'Gladbach', 'FC Koln':'Köln', 'Bayer Leverkusen':'Leverkusen',
    'Paderborn':'Paderborn 07', 'VfB Stuttgart':'Stuttgart',
}
CLUB_KEYS = {normalized(key):normalized(value) for key,value in CLUB_ALIASES.items()}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def club_key(name):
    key = normalized(str(name))
    return CLUB_KEYS.get(key, key)


def country_key(value):
    return {'eng':'gb-eng','sct':'gb-sct','wls':'gb-wls','nir':'gb-nir'}.get(value, value)


def input_files():
    manifest = json.loads((ROOT/'whoscored_enrichment/collection_manifest.json').read_text())
    return [ROOT/'players_data-2026_2027.csv', ROOT/'data/players_advanced_whoscored_2026_2027_totals.csv',
            ROOT/'data/players_advanced_whoscored_2026_2027_per90.csv',
            *sorted((ROOT/'data/position_comparison_sources').glob('*_baseline.json')),
            *[Path(directory)/'raw_capture.json' for directory in manifest['directories']]]


def prepare_data():
    """Baseline and advanced rates must reconcile on player ID, club, league, MP and Min."""
    OUT.mkdir(exist_ok=True)
    raw = pd.read_csv(ROOT/'players_data-2026_2027.csv')
    advanced = pd.read_csv(ROOT/'data/players_advanced_whoscored_2026_2027_per90.csv')
    if len(list((ROOT/'data/position_comparison_sources').glob('*_baseline.json'))) != 5:
        raise ValueError('Collect current baseline source totals for all five leagues first.')
    records, aliases, incompatible = {}, defaultdict(set), set()
    for path in sorted((ROOT/'data/position_comparison_sources').glob('*_baseline.json')):
        capture = json.loads(path.read_text())
        if capture['status'] != 'complete' or capture['observed_season'] != '2026/2027':
            raise ValueError(f'Unverified source: {path}')
        for page in capture['pages']:
            if page['units'] != 'total': raise ValueError('Baseline source must contain totals.')
            for item in page['records']:
                entity = (int(item['playerId']), int(item['teamId']), capture['competition'])
                name_aliases = {item['name'], f'{item.get("firstName", "")} {item.get("lastName", "")}'.strip()}
                rec = records.setdefault(entity, {
                    'source_player_id':entity[0], 'source_team_id':entity[1], 'Comp':entity[2],
                    'WS_Player':item['name'], 'WS_Squad':item['teamName'], 'MP':item['apps'],
                    'Min':item['minsPlayed'], 'Age':item.get('age'), 'country':item.get('regionCode'),
                })
                if (rec['MP'], rec['Min']) != (item['apps'], item['minsPlayed']):
                    incompatible.add(entity)
                for alias in name_aliases:
                    if alias:
                        aliases[(normalized(alias),club_key(item['teamName']),entity[2])].add(entity)
                for name,(category,fields) in BASE_MAPPING.items():
                    if page['category'] != category: continue
                    if not set(fields) <= set(page['stat_columns']):
                        raise ValueError(f'{category}: missing explicitly mapped source fields {fields}')
                    values = [item.get(field) for field in fields]
                    rec[name+'_p90'] = (90*sum(values)/rec['Min']
                        if rec['Min'] > 0 and all(value is not None for value in values) else np.nan)
    # The old labels are metadata. No old performance counters enter this experiment.
    old_eligible = raw.Pos.notna() & ~raw.Pos.str.contains('GK',na=False) & raw['90s'].ge(3)
    accepted, audit = [], []
    used_entities = set()
    advanced_by_key = {(int(row.source_player_id),club_key(row.Squad),row.Comp):row
                       for row in advanced.itertuples(index=False)}
    for row_id,row in raw.iterrows():
        if not old_eligible.loc[row_id]: continue
        candidates = aliases.get((normalized(str(row.Player)),club_key(row.Squad),row.Comp),set())
        status = 'unmatched_identity'
        data = None
        if len(candidates)>1: status='ambiguous_identity'
        elif len(candidates)==1:
            entity = next(iter(candidates)); current = records[entity]
            advanced_row = advanced_by_key.get((entity[0],club_key(current['WS_Squad']),entity[2]))
            positions = list(dict.fromkeys(p.strip() for p in str(row.Pos).split(',') if p.strip() in CLASSES))
            nation = country_key(str(row.Nation).split()[0].lower())
            if entity in incompatible: status='baseline_changed_during_collection'
            elif current['country'] and nation != country_key(current['country']): status='nationality_conflict'
            elif pd.notna(row.Age) and current['Age'] is not None and abs(float(row.Age)-float(current['Age']))>1: status='age_conflict'
            elif advanced_row is None: status='advanced_identity_unavailable'
            elif (current['MP'],current['Min']) != (advanced_row.MP,advanced_row.Min): status='current_source_exposure_mismatch'
            elif current['Min']<270: status='fewer_than_270_current_minutes'
            elif not positions: status='no_valid_position_label'
            elif entity in used_entities: status='repeated_official_identity'
            else:
                status='included'; used_entities.add(entity)
                data = {**current, 'official_source_row':row_id, 'Player':row.Player, 'Squad':row.Squad,
                        'Pos':row.Pos,'primary_label':positions[0], 'pos_set':positions}
                data.update({name:getattr(advanced_row,name) for name in ADVANCED_FEATURES})
                accepted.append(data)
        audit.append({'official_source_row':row_id, 'Player':row.Player,'Squad':row.Squad,'Comp':row.Comp,
                      'Pos':row.Pos,'status':status})
    frame = pd.DataFrame(accepted).reset_index(drop=True)
    if len(frame)<300: raise ValueError(f'Only {len(frame)} verified records; investigate source alignment.')
    audit = pd.DataFrame(audit)
    if frame[BASE_FEATURES+ADVANCED_FEATURES].isna().all().any():
        raise ValueError('A predictor is completely unavailable; do not silently fabricate a column.')
    frame.to_csv(OUT/'comparison_dataset.csv',index=False)
    audit.to_csv(OUT/'cohort_match_audit.csv',index=False)
    primary_counts = frame.primary_label.value_counts().reindex(CLASSES,fill_value=0)
    report = {
        'old_notebook_eligible':int(old_eligible.sum()), 'comparison_records':len(frame),
        'unique_players':int(frame.source_player_id.nunique()),
        'multi_position_records':int(frame.pos_set.str.len().gt(1).sum()),
        'source_filters':'Old labelled outfield cohort (90s >= 3), plus current Min >= 270',
        'match_rule':'Unique exact normalized name alias, explicit club alias and league; nationality and age corroboration; current MP and Min must agree',
        'match_status_counts':audit.status.value_counts().to_dict(), 'primary_class_counts':primary_counts.to_dict(),
        'baseline_features':BASE_FEATURES,'advanced_features':ADVANCED_FEATURES,
        'inputs':{str(path.relative_to(ROOT)):sha(path) for path in input_files()},
        'period_note':'Both numeric feature blocks use reconciled current WhoScored exposure. Official position labels are retained from the earlier CSV. No old numeric performance counters are used.',
        'old_outputs_directly_comparable':False,
    }
    (OUT/'cohort_manifest.json').write_text(json.dumps(report,indent=2))
    return frame, report


def targets(frame):
    valid = np.array([[name in positions for name in CLASSES] for positions in frame.pos_set],dtype=bool)
    return valid, valid/valid.sum(axis=1,keepdims=True)


def model_pipeline(kind):
    model = {'linear':lambda:LinearRegression(),
             'logistic':lambda:LogisticRegression(max_iter=2000),
             'neural':lambda:MLPClassifier(hidden_layer_sizes=(32,16),alpha=1e-2,max_iter=1500,
                                           random_state=42,early_stopping=False)}[kind]()
    return make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),model)


def fit_predict(kind,train,test,features):
    model = model_pipeline(kind)
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
        warnings.simplefilter('always',ConvergenceWarning)
        if kind=='linear':
            _,y=targets(train);model.fit(train[features],y)
            probabilities=np.clip(model.predict(test[features]),0,None)
            totals=probabilities.sum(axis=1,keepdims=True)
            probabilities=np.divide(probabilities,totals,out=np.full_like(probabilities,1/3),where=totals>0)
        else:
            expanded=train.explode('pos_set')
            y=expanded.pos_set.map({name:i for i,name in enumerate(CLASSES)})
            if kind=='logistic':
                weight=1/train.pos_set.str.len().loc[expanded.index].to_numpy()
                model.fit(expanded[features],y,logisticregression__sample_weight=weight)
            else:
                # Match the existing neural notebook's duplicated-label training.
                model.fit(expanded[features],y)
            probabilities=model.predict_proba(test[features])
        convergence=sum(issubclass(w.category,ConvergenceWarning) for w in caught)
    return probabilities,convergence


def score(frame,probabilities):
    valid,soft=targets(frame);pred=probabilities.argmax(axis=1)
    correct=valid[np.arange(len(frame)),pred]; single=valid.sum(axis=1)==1
    return {'any_position_accuracy':float(correct.mean()),
            'primary_accuracy':float(np.mean(np.array(CLASSES)[pred]==frame.primary_label.to_numpy())),
            'single_position_accuracy':float(correct[single].mean()) if single.any() else np.nan,
            'mix_mae':float(mean_absolute_error(soft,probabilities))}


def splits(frame):
    cv=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42)
    folds=list(cv.split(frame,frame.primary_label,groups=frame.source_player_id))
    holdout=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42).split(frame,groups=frame.source_player_id))
    for train,test in [*folds,holdout]:
        assert not set(frame.iloc[train].source_player_id)&set(frame.iloc[test].source_player_id)
    return folds,holdout


def run_experiment(kind,force=False):
    frame,cohort=prepare_data()
    folder=OUT/kind;folder.mkdir(exist_ok=True)
    signature=hashlib.sha256(json.dumps({'inputs':cohort['inputs'],'code':sha(__file__),'kind':kind},sort_keys=True).encode()).hexdigest()
    result_path=folder/'results.json'
    if result_path.exists() and not force:
        result=json.loads(result_path.read_text())
        if result.get('signature')==signature:return frame,cohort,result
    folds,holdout=splits(frame)
    fold_rows,results,predictions=[],{},[]
    for arm,features in [('Baseline',BASE_FEATURES),('Baseline + advanced',BASE_FEATURES+ADVANCED_FEATURES)]:
        oof=np.full((len(frame),3),np.nan);convergence=0
        for fold,(train,test) in enumerate(folds,1):
            print(f'{kind}: {arm}, fold {fold}/5',flush=True)
            p,warn=fit_predict(kind,frame.iloc[train],frame.iloc[test],features)
            oof[test]=p;convergence+=warn
            fold_rows.append({'arm':arm,'fold':fold,'n_test':len(test),**score(frame.iloc[test],p)})
        train,test=holdout
        p,warn=fit_predict(kind,frame.iloc[train],frame.iloc[test],features);convergence+=warn
        results[arm]={'n_features':len(features),'n_holdout':len(test),'cv':score(frame,oof),
                      'holdout':score(frame.iloc[test],p),'convergence_warnings':convergence}
        predictions.append(pd.DataFrame({'source_player_id':frame.source_player_id,'arm':arm,
            'p_DF':oof[:,0],'p_MF':oof[:,1],'p_FW':oof[:,2],'primary_label':frame.primary_label}))
    result={'signature':signature,'model':kind,'cohort_records':len(frame),'arms':results,
            'versions':{name:importlib.metadata.version(name) for name in ['scikit-learn','numpy','pandas']},
            'split':{'cv':'5-fold StratifiedGroupKFold, primary label stratification, player ID grouping, seed 42',
                     'holdout':'20% of player groups, seed 42'},'tuning':'None; predictors fixed before evaluation.'}
    pd.DataFrame(fold_rows).to_csv(folder/'fold_scores.csv',index=False)
    pd.concat(predictions,ignore_index=True).to_csv(folder/'out_of_fold_predictions.csv',index=False)
    result_path.write_text(json.dumps(result,indent=2))
    return frame,cohort,result


def summary_table(result):
    rows=[]
    for arm,values in result['arms'].items():
        rows.append({'Predictors':arm,'Features':values['n_features'],
                     '5-fold accuracy':values['cv']['any_position_accuracy'],
                     'Holdout accuracy':values['holdout']['any_position_accuracy'],
                     'Single-position CV accuracy':values['cv']['single_position_accuracy'],
                     'First-listed CV accuracy':values['cv']['primary_accuracy']})
    return pd.DataFrame(rows).set_index('Predictors')


def conclusion(result):
    base=result['arms']['Baseline'];new=result['arms']['Baseline + advanced']
    delta=100*(new['cv']['any_position_accuracy']-base['cv']['any_position_accuracy'])
    hold=100*(new['holdout']['any_position_accuracy']-base['holdout']['any_position_accuracy'])
    verdict='improved' if delta>0 and hold>0 else 'did not consistently improve'
    return (f'**The advanced predictors {verdict} accuracy in this experiment.** '
            f'Five-fold accuracy changed from **{base["cv"]["any_position_accuracy"]:.1%}** to '
            f'**{new["cv"]["any_position_accuracy"]:.1%}** (**{delta:+.1f} percentage points**); '
            f'the matched holdout changed by **{hold:+.1f} points**. '
            'This compares the same players, folds, source exposure and model configuration. '
            'It is an empirical comparison on this matched cohort, not a claim of future-season performance. '
            'Any-listed-position scoring is more permissive for dual-position players; the single-position '
            'and first-listed figures above show that distinction.')
