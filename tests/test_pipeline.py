import json
import tempfile
import unittest
from pathlib import Path

from cbrc.adapter import verified_assets
from cbrc.statistics import summarize
from cbrc.pipeline import fit


class ProtocolTests(unittest.TestCase):
    def test_missing_assets_do_not_fall_back_to_debug(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            spec={k:str(p/k) for k in ('checkpoint','data_dir','valid_cache','test_cache')}
            with self.assertRaises(FileNotFoundError): verified_assets(spec,p,'valid')

    def test_test_export_cannot_fit(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'export.json').write_text(json.dumps(dict(schema_version=1,split='test',shards=[])))
            with self.assertRaises(ValueError): fit(p,p/'fit')

    def test_smoke_results_cannot_enter_formal_summary(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);r=p/'evaluation.json';r.write_text(json.dumps(dict(smoke=True)))
            with self.assertRaises(ValueError): summarize([r],p/'stats')

    def test_complete_matrix_adoption_and_negative_cell(self):
        # Artificial fixtures exist only in a temporary test directory.
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);paths=[]
            methods=('graph','legacy','fixed_after','cbrc','graph_control','positive','no_semantic','no_graph','no_kl')
            for dataset in ('ICEWS14','YAGO'):
                for k in (5,10):
                    for seed in range(42,52):
                        n=14742 if dataset=='ICEWS14' else 40052
                        metrics={name:dict(mrr=.32 if name=='cbrc' else .31,hits1=.2,hits3=.4,hits10=.5,
                            queries=n,improved=100,unchanged=n-120,damaged=20,
                            positive_rr_contribution=.02,negative_rr_contribution=-.01,
                            correction_seconds_on_exported_scores=.1) for name in methods}
                        result=dict(dataset=dataset,K=k,seed=seed,smoke=False,queries=n,
                            methods=metrics,delta_mrr_cbrc_minus_legacy=.32-.31)
                        p=root/f'{dataset}_{k}_{seed}.json';p.write_text(json.dumps(result));paths.append(p)
            self.assertTrue(summarize(paths,root/'positive')['adopt_module'])
            self.assertTrue((root/'positive/performance_summary.csv').is_file())
            for p in paths:
                row=json.loads(p.read_text())
                if row['dataset']=='YAGO' and row['K']==10:
                    row['methods']['cbrc']['mrr']=.30;row['delta_mrr_cbrc_minus_legacy']=.30-.31
                    p.write_text(json.dumps(row))
            self.assertFalse(summarize(paths,root/'negative')['adopt_module'])


if __name__=='__main__': unittest.main()
