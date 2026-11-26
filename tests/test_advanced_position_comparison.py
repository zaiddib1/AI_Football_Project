"""Guard append-only edits, target isolation and player-level evaluation splits."""
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT));sys.path.insert(0,str(PROJECT.parent/'scripts'))
from advanced_position_comparison import BASE_FEATURES,ADVANCED_FEATURES,splits,targets,score
from append_advanced_position_sections import append_cells_bytes


class AdvancedPositionTests(unittest.TestCase):
    def test_append_preserves_original_bytes_cells_outputs_and_metadata(self):
        old={'cells':[{'cell_type':'code','source':['print("Mbappé")\n'],
                       'execution_count':17,'metadata':{'user_note':'keep'},
                       'outputs':[{'output_type':'stream','name':'stdout','text':['unchanged\n']}]}],
             'metadata':{'custom':'preserve me'},'nbformat':4,'nbformat_minor':5}
        before=(json.dumps(old,ensure_ascii=False,indent=3)+'\n').encode('utf-8')
        new=[{'cell_type':'markdown','metadata':{},'source':['Appended section']}]
        after,start,length=append_cells_bytes(before,new)
        self.assertEqual(after[:start]+after[start+length:],before)
        loaded=json.loads(after)
        self.assertEqual(loaded['cells'][:1],old['cells'])
        self.assertEqual(loaded['metadata'],old['metadata'])

    def test_same_player_never_crosses_train_validation_or_holdout(self):
        frame=pd.DataFrame({'source_player_id':np.repeat(np.arange(45),2),
                            'primary_label':np.repeat(np.array(['DF','MF','FW']*15),2)})
        folds,hold=splits(frame)
        covered=[]
        for train,test in [*folds,hold]:
            self.assertFalse(set(frame.iloc[train].source_player_id)&set(frame.iloc[test].source_player_id))
        for _,test in folds:covered.extend(test)
        self.assertEqual(sorted(covered),list(range(len(frame))))

    def test_alternative_position_targets_are_scored_without_relabelling_truth(self):
        frame=pd.DataFrame({'pos_set':[['DF'],['MF','FW']], 'primary_label':['DF','MF']})
        valid,soft=targets(frame)
        np.testing.assert_allclose(soft,[[1,0,0],[0,.5,.5]])
        observed=score(frame,np.array([[.9,.1,0],[0,.1,.9]]))
        self.assertEqual(observed['any_position_accuracy'],1)
        self.assertEqual(observed['primary_accuracy'],.5)

    def test_predictors_exclude_labels_identifiers_and_ratings(self):
        forbidden={'Pos','Target','primary_label','pos_set','source_player_id','source_team_id',
                   'Player','Squad','Comp','country','rating','playedPositions','positionText'}
        self.assertFalse(forbidden&set(BASE_FEATURES+ADVANCED_FEATURES))
        self.assertEqual(len(BASE_FEATURES),12)
        self.assertEqual(len(ADVANCED_FEATURES),11)
        self.assertEqual(len(set(BASE_FEATURES+ADVANCED_FEATURES)),23)


if __name__=='__main__':unittest.main()
