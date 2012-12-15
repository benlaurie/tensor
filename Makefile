CXXFLAGS=-g -Wall -Werror

all:: .depend test trg-s3

.depend: *.cc *.h
	$(CXX) $(CXXFLAGS) -MM -MG *.cc > .depend

trg-s3: trg-s3.o
	$(CXX) -o trg-s3 trg-s3.o -l gsl -l gslcblas
