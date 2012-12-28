def Args(rank):
    s = ''
    for n in range(rank):
        if n != 0:
            s += ', '
        s += 'uint8_t c' + str(n)
    return s

def Coords(rank):
    print '    uint8_t c[' + str(rank) + '];'
    for n in range(rank):
        print '    c[' + str(n) + '] = c' + str(n) + ';'
    
    
def DTensor(rank):
    print 'class DTensor' + str(rank) + ' : public Tensor<' + str(rank) \
          + ', double> {'
    print ' public:'
    
    print '  void Set(' + Args(rank) + ', const double &value) {'
    Coords(rank)
    print '    Tensor<' + str(rank) + ', double>::Set(c, value);'
    print '  }'

    print '  const double &Get(' + Args(rank) + ') const {'
    Coords(rank)
    print '    return Tensor<' + str(rank) + ', double>::Get(c);'
    print '  }'

    print '  gsl_matrix *GetGSLMatrix(rank_t c1, rank_t c2, rank_t mrow, rank_t mcol,'
    print '      rank_t mrow_size, rank_t mcol_size) {'
    print '    // c1 = value of first "unmatrixed" index'
    print '    // c2 = value of second "unmatrixed" index'
    print '    // mrow = index to be matrix row'
    print '    // mcol = index to be matrix column'
    print '    // mrow_size = number of rows'
    print '    // mcol_size = number of columns'
    print '    uint8_t c[2];'
    print '    c[0] = c1;'
    print '    c[1] = c2;'
    print '    return Tensor<' + str(rank) + ', double>::GetGSLMatrix(c, mrow, mcol, mrow_size, mcol_size);'
    print '  }'

    print '};'
    print

for n in range(1, 26):
    DTensor(n)