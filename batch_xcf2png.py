import os

def do(folder):
    os.chdir(folder)

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
        im_name = fname[:fname.rindex('.')]
        cmd2 = 'convert %s-1.png -background white -alpha remove %s-1.png'%(im_name, im_name)
        os.system(cmd2)


if __name__ == "__main__":
    #do('./opium')
    do('./taliban')
