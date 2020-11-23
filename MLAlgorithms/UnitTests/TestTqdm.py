from tqdm import tqdm
import unittest
import os 

class TestTqdm(unittest.TestCase):

    def test_tqdm(self):
        for i in tqdm(range(1000000)):
            pass
    

if __name__ == '__main__':
    # Grabs the location of the unit test file and sets cwd
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    
    unittest.main()