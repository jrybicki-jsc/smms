import unittest
import pandas as pd
from dataprep import mysample, partition_dataframe



class DataPrepTests(unittest.TestCase):
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
