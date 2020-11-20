import unittest
from appknn import *
import pandas as pd
from numpy.linalg import norm


class SomeTests(unittest.TestCase):
   
    def test_adf(self):
        funcs = {0: {1, 2, 3}, 1: {1}, 2:{2, 4, 6} }
        self.assertEqual(0, adf(1,1, funcs=funcs))
        self.assertEqual(0, adf(0,0, funcs=funcs))
        self.assertEqual(adf(0,1, funcs=funcs), adf(1,0, funcs=funcs))
        self.assertLess(adf(0,1, funcs=funcs), adf(0,2, funcs=funcs)+adf(2,1, funcs=funcs))


    def test_app_k_n(self):
        pts = list(range(1,5))
        new_app = 2.7
        r = app_k_nearest(k=1, apps=pts, new_app=new_app, distance=lambda x,y: norm(x-y))
        self.assertEquals(r[0], 3)

        r2 = app_k_nearest(k=2, apps=pts, new_app=new_app, distance=lambda x,y: norm(x-y))
        self.assertIn(3, r2)
        self.assertIn(2, r2)
    
    def test_create_agg_net(self):
        pts = list(range(1,5)) +[0.3, 2.3]
        net = create_aggregating_net(gamma=0.5, apns=pts, distance=lambda x,y: norm(x-y))
        self.assertEquals(len(net), 5)
        self.assertIn(2, net)
        self.assertIn(2.3, net[2], f"oops {net}")

    def test_jaccar(self):
        a = set([1,2])
        b = set([7, 8, 1])

        res = jaccard(0, 1, {0: a, 1: b})
        self.assertEquals(res, 1-1.0/4)

    def test_classifier(self):
        labels = [0, 1, 0, 1, 0]
        res = lcl(0, labels)
        self.assertEquals(res, [0, 1])

        res = lcl(1, labels)
        self.assertEquals(res, [1, 0])

    def test_create_voting_net(self):
        pts = list(range(1,5)) +[0.3, 2.3]
        vn = create_voting_net(gamma=.5, 
            apns=pts, distance=lambda x,y: norm(x-y), classifier=lambda x: [0,1])
        self.assertEquals(len(vn), 5)
        self.assertIn(2, vn)
        self.assertEquals([0,2], vn[2], f"{vn}")
        self.assertEquals([0,1], vn[1], f"{vn}")

    
    def test_vote(self):
        maj = vote(votes=[0,1])
        self.assertTrue(maj)

        maj = vote(votes=[10,1])
        self.assertFalse(maj)
        
    def test_classify_using_voting(self):
        net = {
            0: [0, 1],
            0.5: [0, 2],
            3: [1,0]
        }
        res = classify_using_voting(app=0, net=net, distance=lambda x,y: norm(x-y), k=1)
        self.assertListEqual([0, 1], list(res))

        res = classify_using_voting(app=0, net=net, distance=lambda x,y: norm(x-y), k=2)
        self.assertListEqual([0, 3], list(res))

        res = classify_using_voting(app=3, net=net, distance=lambda x,y: norm(x-y), k=1)
        self.assertListEqual([1, 0], list(res))

        res = classify_using_voting(app=0, net=net, distance=lambda x,y: norm(x-y), k=3)
        self.assertListEqual([1, 3], list(res))

    def test_evaluate_voting_net(self):
        net = {
            0: [0, 1],
            0.5: [0, 3],
            3: [2,0]
        }
        labels = [0, 0, 1, 1,0,0,0,1]
        distance = lambda  x,y: norm(x-y)
        classifier = lambda x: lcl(x,labels)

        res = evaluate_voting_net(apns=[0], net=net, distance=distance, classifier=classifier, k=1)
        self.assertListEqual(list(res), [0, 0])

        #malicious 3 overvoted by benign 0.5: false postive
        res = evaluate_voting_net(apns=[3], net=net, distance=distance, classifier=classifier, k=2)
        self.assertListEqual(list(res), [0, 1], f"{res}")

    def test_eval_net(self):
        net = {
            0: [0, 1],
            1: [0, 3],
            3: [2,0]
        }
        labels = [0, 1, 1, 0, 1]
        distance = lambda  x,y: norm(x-y)
        classifier = lambda x: lcl(x,labels)

        #TP, FP, TN, FN
        res = eval_net(net=net, test_set=[0, 4], distance=distance, classifier=classifier)
        self.assertEqual(res, (1,0, 1,0), f"{res}")
        
        res = eval_net(net=net, test_set=[1, 3], distance=distance, classifier=classifier)
        self.assertEqual(res, (0,1, 0,1), f"{res}")




    def test_calculate_metrics(self):
        predictions = [True, True, False, False]
        true_values = [True, False, True, False]
        t= calculate_metrics(predictions, true_values)
        self.assertEqual(4, sum(t))
        self.assertTupleEqual(t, (1, 1, 1, 1))

    def test_calculate_net_compression(self):
        net = {
            0: [0, 1],
            0.5: [0, 3],
            3: [2,0]
        }
        res = calculate_net_compression(net)
        self.assertEqual(.5, res)

