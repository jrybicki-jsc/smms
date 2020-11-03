import unittest
from appknn import mysample, partition_dataframe, adf, app_k_nearest, create_aggregating_net, jaccard
import pandas as pd
from numpy.linalg import norm


class SomeTests(unittest.TestCase):
    def test_sample(self):
        df = pd.DataFrame(
            data={
                'apn': list(range(100)),
                'nf': list(range(100))
            })
        ret = mysample(v=df, sample_size=10)
        self.assertEqual(len(ret), 10)

    def test_parts(self):
        df = pd.DataFrame(
            data={
                'apn': list(range(100)),
                'nf': list(range(100))
            })

        n_parts = 2
        ptrs = partition_dataframe(df=df, n_parts=n_parts)
        self.assertEqual(n_parts, len(ptrs))

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


