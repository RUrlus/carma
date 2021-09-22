import build.pymod as pymod

def test_mre():
    d = 30
    mh1 = pymod.MH(d)
    mh2 = pymod.MH(d)
    mh1.A = mh2.A
