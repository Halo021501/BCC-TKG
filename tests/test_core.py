import math
import tempfile
import unittest
from pathlib import Path

import torch

from cbrc.core import (CandidateCorrector,CausalCandidateHistory,PublicQuery,average_rank,
    candidate_features,chronological_partitions,corrected_scores,sparse_objective)
from cbrc.statistics import exact_signed_rank,holm,summarize,paired_statistics

torch.set_num_threads(1)


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.base=torch.log_softmax(torch.randn(2,8),dim=1)
        self.ids=torch.tensor([[1,3,5],[2,6,0]])
        self.mask=torch.tensor([[True,True,True],[True,True,False]])
        self.z=torch.rand(2,3,10)

    def test_parameter_count_identity_and_empty_fallback(self):
        model=CandidateCorrector()
        self.assertEqual(sum(p.numel() for p in model.parameters()),193)
        d=model(self.z,self.mask)
        self.assertTrue(torch.equal(self.base,corrected_scores(self.base,self.ids,self.mask,d)))
        with torch.no_grad(): model.net[-1].bias.fill_(10)
        empty=torch.zeros_like(self.mask)
        self.assertTrue(torch.equal(self.base,corrected_scores(self.base,self.ids,empty,model(self.z,empty))))
        self.assertTrue(torch.equal(self.base,corrected_scores(self.base,self.ids,self.mask,model(self.z,self.mask,0))))

    def test_bounds_and_permutation(self):
        model=CandidateCorrector()
        with torch.no_grad(): model.net[-1].weight.normal_(0,5)
        delta=model(self.z,self.mask)
        self.assertLessEqual(delta.abs().max(),.35)
        perm=torch.tensor([2,0,1])
        a=corrected_scores(self.base,self.ids,self.mask,delta)
        b=corrected_scores(self.base,self.ids[:,perm],self.mask[:,perm],model(self.z[:,perm],self.mask[:,perm]))
        self.assertTrue(torch.equal(a,b))
        shifts=a-self.base
        self.assertLessEqual((shifts[:,:,None]-shifts[:,None,:]).abs().max(),.700001)

    def test_full_vocab_sparse_loss_and_gradient_match(self):
        delta=(torch.rand(2,3,dtype=torch.float64)*.5-.25).requires_grad_()
        truths=[[1,4],[3,6]]
        loss,_=sparse_objective(self.base,self.ids,self.mask,delta,truths)
        base=self.base.double()
        dense=corrected_scores(base,self.ids,self.mask,delta)
        logp=dense.log_softmax(1)
        nll=-torch.stack([logp[i,t].logsumexp(0) for i,t in enumerate(truths)]).mean()
        kl=(base.exp()*(base-logp)).sum(1).mean()
        self.assertAlmostEqual(float(loss),float(nll+kl),places=6)
        g1=torch.autograd.grad(loss,delta,retain_graph=True)[0]
        g2=torch.autograd.grad(nll+kl,delta)[0]
        self.assertTrue(torch.allclose(g1,g2,atol=1e-7))

    def test_public_query_rejects_gold(self):
        with self.assertRaises(ValueError):
            PublicQuery.from_mapping(dict(known_entity_id=1,oriented_relation_id=0,timestamp=2,target=3))

    def test_features_are_unfiltered_and_strictly_past(self):
        h=CausalCandidateHistory([(0,0,1,1),(0,0,1,4),(0,0,1,5),(0,0,1,8)])
        h.begin_timestamp(5)
        q=[PublicQuery(0,0,5)]
        base=torch.tensor([[-2.,-1.,-1.,-3.]])
        ids=torch.tensor([[1,2]])
        z=candidate_features(base,q,ids,torch.ones_like(ids,dtype=torch.bool),torch.zeros(1,2,5),h)
        self.assertAlmostEqual(float(z[0,0,8]),math.log(3)/math.log(11),places=6)
        self.assertAlmostEqual(float(z[0,0,6]),math.log1p(1.5)/math.log(5),places=6)
        self.assertEqual(float(z[0,1,8]),0.)
        with self.assertRaises(ValueError): h.begin_timestamp(6)
        with self.assertRaises(ValueError): h.commit_timestamp([(0,0,2,6)])
        h.commit_timestamp([(0,0,2,5)])
        h.begin_timestamp(6)
        self.assertGreater(h.features(PublicQuery(0,0,6),[2])[0][0],0)

    def test_duplicate_and_invalid_mapping_rejected(self):
        with self.assertRaises(ValueError):
            corrected_scores(self.base,torch.tensor([[1,1,2],[1,2,3]]),self.mask,torch.zeros(2,3))
        with self.assertRaises(ValueError):
            corrected_scores(self.base,torch.full((2,3),99),self.mask,torch.zeros(2,3))

    def test_partition_whole_timestamps(self):
        p=chronological_partitions([i//3 for i in range(30)])
        self.assertEqual([p[i] for i in range(10)],['fit']*6+['select']*2+['audit']*2)
        with self.assertRaises(ValueError): chronological_partitions([0,1])

    def test_average_ties_and_multitruth(self):
        self.assertEqual(average_rank(torch.tensor([2.,2.,1.]),0,[]),1.5)
        self.assertEqual(average_rank(torch.tensor([2.,2.,1.]),0,[0,1]),1.)
        self.assertEqual(average_rank(torch.ones(8),4,[]),4.5)

    def test_positive_ablation_trains_from_zero(self):
        model=CandidateCorrector('positive')
        d=model(self.z,self.mask)
        loss,_=sparse_objective(self.base,self.ids,self.mask,d,[[1],[2]])
        loss.backward()
        self.assertGreater(float(model.net[-1].weight.grad.abs().sum()),0.)

    def test_small_exact_tests_and_incomplete_refusal(self):
        self.assertEqual(exact_signed_rank([1.]*10),2/1024)
        self.assertEqual(exact_signed_rank([0.]*10),1.)
        self.assertEqual(holm([.001953125]*4),[.0078125]*4)
        with tempfile.TemporaryDirectory() as p:
            self.assertFalse(summarize([],p)['adopt_module'])

    def test_extreme_target_probability_and_empty_candidates(self):
        base=torch.tensor([[0.,-10000.,-2.]])
        ids=torch.empty(1,0,dtype=torch.long);mask=torch.empty(1,0,dtype=torch.bool)
        d=torch.empty(1,0,requires_grad=True)
        loss,parts=sparse_objective(base,ids,mask,d,[[1]])
        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(float(parts['kl']),0.)

    def test_feature_ablation_masks(self):
        for variant,sl in [('no_semantic',slice(0,5)),('graph_control',slice(0,5)),('no_graph',slice(5,10))]:
            m=CandidateCorrector(variant)
            with torch.no_grad(): m.net[-1].weight.fill_(.1)
            z=self.z.clone();z[...,sl]+=5
            self.assertTrue(torch.equal(m(z,self.mask),m(self.z,self.mask)))

    def test_forward_cannot_backpropagate_into_graph_features(self):
        z=self.z.clone().requires_grad_();m=CandidateCorrector()
        with torch.no_grad():m.net[-1].weight.fill_(.1)
        m(z,self.mask).sum().backward()
        self.assertIsNone(z.grad)

    def test_bootstrap_reproducible_and_negative_effect(self):
        a=paired_statistics([-.001*i for i in range(1,11)])
        self.assertEqual(a,paired_statistics([-.001*i for i in range(1,11)]))
        self.assertLess(a['ci95_high_pp'],0.)


if __name__=='__main__': unittest.main()
