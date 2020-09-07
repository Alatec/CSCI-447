import unittest
import os



if __name__ == '__main__':
    # Grabs the location of the unit test file and sets cwd
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    
    loader = unittest.TestLoader()

    tests = loader.discover(".","Test*.py")
    testRunner = unittest.runner.TextTestRunner()
    results = testRunner.run(tests)

    exit(len(results.errors)+len(results.failures)) 