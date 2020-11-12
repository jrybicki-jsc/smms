import unittest
import pandas as pd
from dataprep import mysample, partition_dataframe, get_part_indexes, r1



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

    def test_get_part_indexes(self):
        df = pd.DataFrame(
            data={
                'apn': list(range(100)),
                'nf': list(range(100))
            })
        ret = get_part_indexes(dfs=df, num_parts=5, size=10)
        self.assertEqual(len(ret), 5)
        self.assertEquals(len(ret[0]), 10)

    def test_metrics(self):
        a = {1: [6, 31, 0, 14], 2: [1, 23, 55, 1]}
        r = pd.DataFrame.from_dict(a, orient='index', columns=['tp', 'fp', 'tn', 'fn'])

        s1, s2 = r.apply(r1, axis=1)
        self.assertAlmostEqual(s1, 0.2105263157894737)
        self.assertAlmostEqual(s2, 0.07692307692307693)



