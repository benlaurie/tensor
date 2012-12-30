/* -*- mode: c++; indent-tabs-mode: nil -*- */

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <gsl/gsl_matrix.h>

#include <iostream>
#include <map>

// Should be unsigned but causes warning "comparison is always false
// due to limited range of data" when rank 0 is used with some
// compilers.
typedef int8_t rank_t;

template <rank_t rank> class Coordinate {
public:
  explicit Coordinate(const uint8_t coords[rank]) {
    memcpy(coords_, coords, sizeof coords_);
  }
  Coordinate() {
    for (rank_t d = 0; d < rank; ++d)
        coords_[d] = 0xff;
  }
  bool operator<(const Coordinate &other) const {
    for (rank_t d = 0; d < rank; ++d)
      if (coords_[d] < other.coords_[d])
        return true;
      else if (coords_[d] > other.coords_[d])
        return false;
    return false;
  }
  void Print(std::ostream &os) const {
    for (rank_t r = 0; r < rank; ++r) {
      if (r != 0)
        os << ", ";
      os << (int)coords_[r];
    }
  }
  template <rank_t rank2> void Set(rank_t offset, Coordinate<rank2> coords) {
    assert(rank2 + offset <= rank);
    for (rank_t r = 0; r < rank2; ++r)
      coords_[r + offset] = coords.coord(r);
  }
  void Set(rank_t r, uint8_t coord) {
    assert(r < rank);
    coords_[r] = coord;
  }
      
  const uint8_t coord(rank_t r) const {
    assert(r < rank);
    return coords_[r];
  }

  const Coordinate<rank - 1> except(rank_t d) const {
    Coordinate<rank - 1> result;
    for (uint8_t r = 0; r < rank; ++r)
      if (r < d)
        result.Set(r, coords_[r]);
      else if (r > d)
        result.Set(r - 1, coords_[r]);
    return result;
  }

  const Coordinate<rank - 2> except2(rank_t d1, uint8_t d2) const {
    Coordinate<rank - 2> result;
    for (rank_t r = 0; r < rank; ++r)
      if (r < d1 && r < d2)
        result.Set(r, coords_[r]);
      else if ((r > d1 && r < d2) || (r < d1 && r > d2))
        result.Set(r - 1, coords_[r]);
      else if (r > d1 && r > d2)
        result.Set(r - 2, coords_[r]);
    return result;
  }

  bool operator==(const Coordinate &rhs) const {
    return memcmp(coords_, rhs.coords_, sizeof coords_) == 0;
  }

private:
  uint8_t coords_[rank];
};

template <rank_t rank> std::ostream &operator<<(std::ostream &out, const Coordinate<rank> &coord) {
  coord.Print(out);
  return out;
} 

template <rank_t rank, class Value> class Tensor {
public:
  typedef std::pair<Coordinate<rank>, Value> EPair;

  Tensor() {}

  // leave undefined
  Tensor(const Tensor &from);
  Tensor &operator=(const Tensor &rhs);

  bool operator==(const Tensor &rhs) const {
    return elements_ == rhs.elements_;
  }

  void Set(uint8_t coords[rank], const Value &value) {
    Coordinate<rank> coord(coords);
    Set(coord, value);
  }
  void Set(Coordinate<rank> coord, const Value &value) {
    assert(!isinf(value));
    assert(!isnan(value));
    if (value == 0) {
      elements_.erase(coord);
      return;
    }
    std::pair<typename std::map<Coordinate<rank>, Value>::iterator, bool> ret
        = elements_.insert(EPair(coord, value));
    if (!ret.second)
      ret.first->second = value;
  }
  const Value &Get(uint8_t coords[rank]) const {
    Coordinate<rank> coord(coords);
    return Get(coord);
  }
  const Value &Get(Coordinate<rank> coord) const {
    typename std::map<Coordinate<rank>, Value>::const_iterator i = elements_.find(coord);
    if (i == elements_.end()) {
      static Value zero(0);
      return zero;
    }
    return i->second;
  }
  void Print(std::ostream &os) const {
    typename std::map<Coordinate<rank>, Value>::const_iterator i;
    for (i = elements_.begin(); i != elements_.end(); ++i) {
      os << '[' << i->first << ": " << i->second << ']';
    }
  }
  const std::map<Coordinate<rank>, Value> &elements() const { return elements_; }
  gsl_matrix* GetGSLMatrix(uint8_t coords[rank - 2], uint8_t mrow, uint8_t mcol,
      uint8_t mrow_size, uint8_t mcol_size) const {
    gsl_matrix *M = gsl_matrix_calloc(mrow_size, mrow_size);
    uint8_t full_coords[rank];
    uint8_t d = 0;
    for (uint8_t i = 0; i < rank; ++i)
      if (i != mrow && i != mcol) {
        full_coords[i] = coords[d];
        ++d;
      }
    for (uint8_t i = 0; i < mrow_size; ++i)
      for (uint8_t j = 0; j < mcol_size; ++j) {
        full_coords[mrow] = i;
        full_coords[mcol] = j;
        gsl_matrix_set(M, i, j, Get(full_coords));
      }
    return M;
  }
private:
  std::map<Coordinate<rank>, Value> elements_;
};

template <rank_t rank, class Value> std::ostream &operator<<(std::ostream &out, const Tensor<rank, Value> &tensor) {
  tensor.Print(out);
  return out;
}

template <rank_t rank1, rank_t rank2, class Value>
void Multiply(Tensor<rank1 + rank2, Value> *t_out,
              const Tensor<rank1, Value> &t1,
              const Tensor<rank2, Value> &t2) {
  typename std::map<Coordinate<rank1>, Value>::const_iterator i1;
  for (i1 = t1.elements().begin(); i1 != t1.elements().end(); ++i1) {
    typename std::map<Coordinate<rank2>, Value>::const_iterator i2;
    for (i2 = t2.elements().begin(); i2 != t2.elements().end(); ++i2) {
      Coordinate<rank1 + rank2> new_coord;
      new_coord.Set(0, i1->first);
      new_coord.Set(rank1, i2->first);
      t_out->Set(new_coord, i1->second * i2->second);
    }
  }
}

template <rank_t rank1, rank_t rank2, class Value>
void Contract(Tensor<rank1 + rank2 - 1, Value> *t_out,
              const Tensor<rank1, Value> &t1, uint8_t d1,
              const Tensor<rank2, Value> &t2, uint8_t d2) {
  typedef std::multimap<uint8_t,
      const std::pair<const Coordinate<rank2>, Value> *> Map;
  Map t2map;
  typename std::map<Coordinate<rank2>, Value>::const_iterator i2;
  for (i2 = t2.elements().begin(); i2 != t2.elements().end(); ++i2)
    t2map.insert(std::pair<uint8_t, const std::pair<const Coordinate<rank2>,
                 Value> *>(i2->first.coord(d2), &*i2));

  typename std::map<Coordinate<rank1>, Value>::const_iterator i1;
  for (i1 = t1.elements().begin(); i1 != t1.elements().end(); ++i1) {
    uint8_t r = i1->first.coord(d1);
    std::pair<typename Map::const_iterator,
        typename Map::const_iterator> r2 = t2map.equal_range(r);
    for (typename Map::const_iterator i2 = r2.first; i2 != r2.second; ++i2) {
      Coordinate<rank1 + rank2 - 1> new_coord;
      new_coord.Set(0, i1->first);
      new_coord.Set(rank1, i2->second->first.except(d2));
      t_out->Set(new_coord, i1->second * i2->second->second);
    }
  }
}

template <rank_t rank1, rank_t rank2, class Value>
void Contract2(Tensor<rank1 + rank2 - 2, Value> *t_out,
               const Tensor<rank1, Value> &t1, uint8_t d1,
               const Tensor<rank2, Value> &t2, uint8_t d2) {
  typedef std::multimap<uint8_t,
      const std::pair<const Coordinate<rank2>, Value> *> Map;
  Map t2map;
  typename std::map<Coordinate<rank2>, Value>::const_iterator i2;
  for (i2 = t2.elements().begin(); i2 != t2.elements().end(); ++i2)
    t2map.insert(std::pair<uint8_t,
        const std::pair<const Coordinate<rank2>, Value> *>(i2->first.coord(d2),
                                                           &*i2));

  typename std::map<Coordinate<rank1>, Value>::const_iterator i1;
  for (i1 = t1.elements().begin(); i1 != t1.elements().end(); ++i1) {
    uint8_t r = i1->first.coord(d1);
    std::pair<typename Map::const_iterator, typename Map::const_iterator> r2
        = t2map.equal_range(r);
    if (r2.first != r2.second) {
      for (typename Map::const_iterator i2 = r2.first; i2 != r2.second; ++i2) {
        Coordinate<rank1 + rank2 - 2> new_coord;
        new_coord.Set(0, i1->first.except(d1));
        new_coord.Set(rank1 - 1, i2->second->first.except(d2));
        t_out->Set(new_coord, t_out->Get(new_coord)
                   + i1->second * i2->second->second);
      }
    }
  }
}

template <rank_t rank, class Value>
void ContractSelfOld(Tensor<rank - 1, Value> *t_out,
              const Tensor<rank, Value> &t_in, uint8_t d1, uint8_t d2) {
  typedef std::multimap<uint8_t,
      const std::pair<const Coordinate<rank>, Value> *> Map;
  Map t_in_map;
  typename std::map<Coordinate<rank>, Value>::const_iterator i2;
  for (i2 = t_in.elements().begin(); i2 != t_in.elements().end(); ++i2)
    t_in_map.insert(std::pair<uint8_t, const std::pair<const Coordinate<rank>,
                 Value> *>(i2->first.coord(d2), &*i2));

  typename std::map<Coordinate<rank>, Value>::const_iterator i1;
  for (i1 = t_in.elements().begin(); i1 != t_in.elements().end(); ++i1) {
    uint8_t r = i1->first.coord(d1);
    std::pair<typename Map::const_iterator,
        typename Map::const_iterator> r2 = t_in_map.equal_range(r);
    for (typename Map::const_iterator i2 = r2.first; i2 != r2.second; ++i2) {
      Coordinate<rank - 1> new_coord;
      new_coord.Set(0, i1->first.except(d2));
      t_out->Set(new_coord, i1->second * i2->second->second);
    }
  }
}

template <rank_t rank, class Value>
void ContractSelf(Tensor<rank - 1, Value> *t_out,
              const Tensor<rank, Value> &t_in, uint8_t d1, uint8_t d2) {
  typename std::map<Coordinate<rank>, Value>::const_iterator i1;
  for (i1 = t_in.elements().begin(); i1 != t_in.elements().end(); ++i1) {
    if (i1->first.coord(d1) == i1->first.coord(d2)) {
      Coordinate<rank - 1> new_coord;
      new_coord.Set(0, i1->first.except(d2));
      t_out->Set(new_coord, i1->second);
    }
  }
}

template <rank_t rank, class Value>
void ContractSelf2old(Tensor<rank - 2, Value> *t_out,
               const Tensor<rank, Value> &t_in, uint8_t d1, uint8_t d2) {
  typedef std::multimap<uint8_t,
      const std::pair<const Coordinate<rank>, Value> *> Map;
  Map t_in_map;
  typename std::map<Coordinate<rank>, Value>::const_iterator i2;
  for (i2 = t_in.elements().begin(); i2 != t_in.elements().end(); ++i2)
    t_in_map.insert(std::pair<uint8_t,
        const std::pair<const Coordinate<rank>, Value> *>(i2->first.coord(d2),
                                                           &*i2));

  typename std::map<Coordinate<rank>, Value>::const_iterator i1;
  for (i1 = t_in.elements().begin(); i1 != t_in.elements().end(); ++i1) {
    uint8_t r = i1->first.coord(d1);
    std::pair<typename Map::const_iterator, typename Map::const_iterator> r2
        = t_in_map.equal_range(r);
    if (r2.first != r2.second) {
      for (typename Map::const_iterator i2 = r2.first; i2 != r2.second; ++i2) {
        Coordinate<rank - 2> new_coord;
        new_coord.Set(0, i1->first.except2(d1, d2));
        t_out->Set(new_coord, t_out->Get(new_coord)
                   + i1->second * i2->second->second);
      }
    }
  }
}

template <rank_t rank, class Value>
void ContractSelf2(Tensor<rank - 2, Value> *t_out,
               const Tensor<rank, Value> &t_in, uint8_t d1, uint8_t d2) {
  typename std::map<Coordinate<rank>, Value>::const_iterator i1;
  for (i1 = t_in.elements().begin(); i1 != t_in.elements().end(); ++i1) {
    if (i1->first.coord(d1) == i1->first.coord(d2)) {
        Coordinate<rank - 2> new_coord;
        new_coord.Set(0, i1->first.except2(d1, d2));
        t_out->Set(new_coord, t_out->Get(new_coord) + i1->second);
    }
  }
}

#if 0
class DTensor1 : public Tensor<1, double> {
public:
  void Set(uint8_t c1, const double &value) {
    uint8_t c[1];
    c[0] = c1;
    Tensor<1, double>::Set(c, value);
  }
};

class DTensor2 : public Tensor<2, double> {
public:
  void Set(uint8_t c1, uint8_t c2, const double &value) {
    uint8_t c[2];
    c[0] = c1;
    c[1] = c2;
    Tensor<2, double>::Set(c, value);
  }
};

class DTensor3 : public Tensor<3, double> {
public:
  void Set(uint8_t c1, uint8_t c2, uint8_t c3, const double &value) {
    uint8_t c[3];
    c[0] = c1;
    c[1] = c2;
    c[2] = c3;
    Tensor<3, double>::Set(c, value);
  }
  const double &Get(uint8_t c1, uint8_t c2, uint8_t c3) const {
    uint8_t c[3];
    c[0] = c1;
    c[1] = c2;
    c[2] = c3;
    return Tensor<3, double>::Get(c);
  }
};

class DTensor4 : public Tensor<4, double> {
public:
  void Set(uint8_t c1, uint8_t c2, uint8_t c3, uint8_t c4,
      const double &value) {
    uint8_t c[4];
    c[0] = c1;
    c[1] = c2;
    c[2] = c3;
    c[3] = c4;
    Tensor<4, double>::Set(c, value);
  }
  const double &Get(uint8_t c1, uint8_t c2, uint8_t c3, uint8_t c4) const {
    uint8_t c[4];
    c[0] = c1;
    c[1] = c2;
    c[2] = c3;
    c[3] = c4;
    return Tensor<4, double>::Get(c);
  }
  gsl_matrix* GetGSLMatrix(uint8_t c1, uint8_t c2, uint8_t mrow, uint8_t mcol,
      uint8_t mrow_size, uint8_t mcol_size) {
    // c1 = value of first "unmatrixed" index
    // c2 = value of second "unmatrixed" index
    // mrow = index to be matrix row
    // mcol = index to be matrix column
    // mrow_size = number of rows
    // mcol_size = number of columns
    uint8_t c[2];
    c[0] = c1;
    c[1] = c2;
    return Tensor<4, double>::GetGSLMatrix(c, mrow, mcol, mrow_size, mcol_size);
  }
};

class DTensor5 : public Tensor<5, double> {
public:
  void Set(uint8_t c1, uint8_t c2, uint8_t c3, uint8_t c4, uint8_t c5,
      const double &value) {
    uint8_t c[5];
    c[0] = c1;
    c[1] = c2;
    c[2] = c3;
    c[3] = c4;
    c[4] = c5;
    Tensor<5, double>::Set(c, value);
  }
  const double &Get(uint8_t c1, uint8_t c2, uint8_t c3, uint8_t c4,
      uint8_t c5) const {
    uint8_t c[5];
    c[0] = c1;
    c[1] = c2;
    c[2] = c3;
    c[3] = c4;
    c[4] = c5;
    return Tensor<5, double>::Get(c);
  }
};

class DTensor9 : public Tensor<9, double> {
public:
  void Set(uint8_t c1, uint8_t c2, uint8_t c3, uint8_t c4, uint8_t c5,
      uint8_t c6, uint8_t c7, uint8_t c8, uint8_t c9, const double &value) {
    uint8_t c[9];
    c[0] = c1;
    c[1] = c2;
    c[2] = c3;
    c[3] = c4;
    c[4] = c5;
    c[5] = c6;
    c[6] = c7;
    c[7] = c8;
    c[8] = c9;
    Tensor<9, double>::Set(c, value);
  }
  const double &Get(uint8_t c1, uint8_t c2, uint8_t c3, uint8_t c4, uint8_t c5,
      uint8_t c6, uint8_t c7, uint8_t c8, uint8_t c9) const {
    uint8_t c[9];
    c[0] = c1;
    c[1] = c2;
    c[2] = c3;
    c[3] = c4;
    c[4] = c5;
    c[5] = c6;
    c[6] = c7;
    c[7] = c8;
    c[8] = c9;
    return Tensor<9, double>::Get(c);
  }
};

class DTensor14 : public Tensor<14, double> {
public:
  void Set(uint8_t c1, uint8_t c2, uint8_t c3, uint8_t c4, uint8_t c5,
      uint8_t c6, uint8_t c7, uint8_t c8, uint8_t c9, uint8_t c10, uint8_t c11,
      uint8_t c12, uint8_t c13, uint8_t c14, const double &value) {
    uint8_t c[14];
    c[0] = c1;
    c[1] = c2;
    c[2] = c3;
    c[3] = c4;
    c[4] = c5;
    c[5] = c6;
    c[6] = c7;
    c[7] = c8;
    c[8] = c9;
    c[9] = c10;
    c[10] = c11;
    c[11] = c12;
    c[12] = c13;
    c[13] = c14;
    Tensor<14, double>::Set(c, value);
  }
  const double &Get(uint8_t c1, uint8_t c2, uint8_t c3, uint8_t c4, uint8_t c5,
      uint8_t c6, uint8_t c7, uint8_t c8, uint8_t c9, uint8_t c10, uint8_t c11,
      uint8_t c12, uint8_t c13, uint8_t c14) const {
    uint8_t c[14];
    c[0] = c1;
    c[1] = c2;
    c[2] = c3;
    c[3] = c4;
    c[4] = c5;
    c[5] = c6;
    c[6] = c7;
    c[7] = c8;
    c[8] = c9;
    c[9] = c10;
    c[10] = c11;
    c[11] = c12;
    c[12] = c13;
    c[13] = c14;
    return Tensor<14, double>::Get(c);
  }
};
#endif
#include "auto_tensor.h"

