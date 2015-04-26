import os

def do():
    flist = os.listdir('./')
    print flist
    for fname in flist:
        if not fname.endswith('xcf'):
            continue
        if os.path.isfile(fname.replace('xcf','png')):
            continue
        cmd = 'convert %s %s'%(fname, fname.replace('xcf','png'))
        print cmd
        os.system(cmd)


if __name__ == "__main__":
    do()
