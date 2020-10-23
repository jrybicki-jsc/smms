import unittest
from appknn import mysample, partition_dataframe, adf
import pandas as pd


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
        #def adf(apid1: int, apid2: int, funcs) -> float:
        funcs = {0: {1, 2, 3}, 1: {1}, 2:{2, 4, 6} }
        self.assertEqual(0, adf(1,1, funcs=funcs))
        self.assertEqual(0, adf(0,0, funcs=funcs))
        self.assertEqual(adf(0,1, funcs=funcs), adf(1,0, funcs=funcs))
        self.assertLess(adf(0,1, funcs=funcs), adf(0,2, funcs=funcs)+adf(2,1, funcs=funcs))
        
        
        

