CXXFLAGS+=-g -Wall -Werror -O3

all:: .depend test trg-s3

-include .depend

.depend: *.cc *.h
	$(CXX) $(CXXFLAGS) -MM -MG *.cc > .depend

auto_tensor.h: autocode.py
	python autocode.py > auto_tensor.h

test: test.o
	$(CXX) $(CXXFLAGS) -o test test.o

trg-s3: trg-s3.o
	$(CXX) $(CXXFLAGS) -o trg-s3 trg-s3.o -l gsl -l gslcblas

clean:
	rm -f *.o test trg-s3 auto_tensor.h
