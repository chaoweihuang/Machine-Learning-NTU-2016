def write_submit_file(l, fname):
    if len(l) != 10000:
        return
    with open(fname, 'w') as fw:
        fw.write('ID,class\n')
        fw.write('\n'.join(['%d,%d' % (i, l[i]) for i in range(10000)]))
