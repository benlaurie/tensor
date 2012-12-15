CXXFLAGS=-g -Wall -Werror

all:: test trg-s3

trg-s3.o: tensor.h

trg-s3: trg-s3.o
	$(CXX) -o trg-s3 trg-s3.o -l gsl -l blas