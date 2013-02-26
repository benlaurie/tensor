/* -*- mode: c++; indent-tabs-mode: nil -*- */

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <gsl/gsl_matrix.h>

#include <iostream>
#include <map>
#include <set>

// Should be unsigned but causes warning "comparison is always false
// due to limited range of data" when rank 0 is used with some
// compilers.
typedef int8_t rank_t;

inline bool EffectivelyZero(double value) {
  return fabs(value) < 1e-12;
}


template <rank_t rank> class Coordinate {
public:
  explicit Coordinate(const uint8_t coords[rank]) {
    memcpy(coords_, coords, sizeof coords_);
  }

  Coordinate(const uint8_t *c1, rank_t d, uint8_t c, const uint8_t *c2) {
    assert(d < rank);
    memcpy(coords_, c1, d);
    coords_[d] = c;
    memcpy(&coords_[d + 1], c2, rank - d - 1);
  }

  Coordinate() {
    //for (rank_t d = 0; d < rank; ++d)
    //  coords_[d] = 0xff;
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

  template <rank_t rank2> void Set(rank_t offset,
                                   const Coordinate<rank2> &coords) {
    assert(rank2 + offset <= rank);
    for (rank_t r = 0; r < rank2; ++r)
      coords_[r + offset] = coords.coord(r);
  }

  template <rank_t rank1, rank_t rank2> void Set(const Coordinate<rank1> &c1,
                                                 const Coordinate<rank2> &c2) {
    assert(rank1 + rank2 == rank);
    for (rank_t r = 0; r < rank1; ++r)
      coords_[r] = c1[r];
    for (rank_t r = 0; r < rank2; ++r)
      coords_[rank1 + r] = c2[r];
  }

  void Set(rank_t r, uint8_t coord) {
    assert(r < rank);
    assert(r > -1);
    coords_[r] = coord;
  }

  uint8_t coord(rank_t r) const {
    assert(r < rank);
    return coords_[r];
  }

  uint8_t operator[](rank_t r) const {
    return coord(r);
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

  const uint8_t *coords() const {
    return coords_;
  }

private:
  uint8_t coords_[rank];
};

template <rank_t rank>
std::ostream &operator<<(std::ostream &out,
                         const Coordinate<rank> &coord) {
  coord.Print(out);
  return out;
}

template <rank_t rank, class Value> class Tensor {
public:
  typedef std::pair<Coordinate<rank>, Value> EPair;
  typedef typename std::map<Coordinate<rank>, Value>::const_iterator Iterator;
  typedef Value ValueType;
  static const rank_t Rank = rank;

  Tensor() {}

  // leave undefined
  Tensor(const Tensor &from);
  Tensor &operator=(const Tensor &rhs);

  // Note that this should override the general-purpose == operator
  // defined below.
  bool operator==(const Tensor &rhs) const {
    return elements_ == rhs.elements_;
  }

  void Set(uint8_t coords[rank], const Value &value) {
    Coordinate<rank> coord(coords);
    Set(coord, value);
  }

  void Set(const Coordinate<rank> &coord, const Value &value) {
    assert(!isinf(value));
    assert(!isnan(value));
    // FIXME: use condi
    if (EffectivelyZero(value)) {
      elements_.erase(coord);
      return;
    }
     
    std::pair<typename std::map<Coordinate<rank>, Value>::iterator, bool> ret
        = elements_.insert(EPair(coord, value));
    if (!ret.second)
      ret.first->second = value;
    else if (elements_.size() == 1)
      for (rank_t r = 0; r < rank; ++r)
        low_[r] = high_[r] = coord[r];
    else
      for (rank_t r = 0; r < rank; ++r)
        if (coord[r] < low_[r])
          low_[r] = coord[r];
        else if (coord[r] > high_[r])
          high_[r] = coord[r];
  }

  const Value &Get(uint8_t coords[rank]) const {
    Coordinate<rank> coord(coords);
    return Get(coord);
  }

  const Value &Get(const Coordinate<rank> &coord) const {
    typename std::map<Coordinate<rank>, Value>::const_iterator i
        = elements_.find(coord);
    if (i == elements_.end()) {
      static Value zero(0);
      return zero;
    }
    return i->second;
  }

  uint8_t Low(uint8_t d) const {
    assert(d < rank);
    return low_[d];
  }
  uint8_t High(uint8_t d) const {
    assert(d < rank);
    return high_[d];
  }

  void Print(std::ostream &os) const {
    typename std::map<Coordinate<rank>, Value>::const_iterator i;
    for (i = elements_.begin(); i != elements_.end(); ++i) {
      os << '[' << i->first << ": " << i->second << ']';
    }
  }

  const std::map<Coordinate<rank>, Value> &elements() const {
    return elements_;
  }

  typename std::map<Coordinate<rank>, Value>::const_iterator begin() const {
    return elements_.begin();
  }

  typename std::map<Coordinate<rank>, Value>::const_iterator end() const {
    return elements_.end();
  }

  // Note: only works on tensors of rank at least 2, unsurprisingly
  gsl_matrix *GetGSLMatrix(uint8_t coords[rank > 1 ? rank - 2 : 1],
                           uint8_t mrow, uint8_t mcol,
                           uint8_t mrow_size, uint8_t mcol_size) const {
    assert(rank > 1);
    gsl_matrix *M = gsl_matrix_calloc(mrow_size, mcol_size);
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

  void Clear() {
    elements_.clear();
  }

private:
  std::map<Coordinate<rank>, Value> elements_;
  uint8_t low_[rank];
  uint8_t high_[rank];
};

template <rank_t rank, class Value>
std::ostream &operator<<(std::ostream &out, const Tensor<rank, Value> &tensor) {
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

//FIXME: rename Contract2 -> Contract and Contract -> something more appropriate

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
    uint8_t r1 = i1->first.coord(d1);
    std::pair<typename Map::const_iterator,
        typename Map::const_iterator> r2 = t2map.equal_range(r1);
    for (typename Map::const_iterator i2 = r2.first; i2 != r2.second; ++i2) {
      Coordinate<rank1 + rank2 - 1> new_coord;
      new_coord.Set(0, i1->first);
      new_coord.Set(rank1, i2->second->first.except(d2));
      t_out->Set(new_coord, i1->second * i2->second->second);
    }
  }
}

template <class Tensor1, rank_t d1, class Tensor2, rank_t d2>
class ContractedTensor {
 public:
  static const rank_t Rank = Tensor1::Rank + Tensor2::Rank - 1;
  typedef typename Tensor1::ValueType ValueType;

  ContractedTensor(const Tensor1 *t1, const Tensor2 *t2)
      : t1_(t1), t2_(t2) {
  }

  const ValueType Get(const uint8_t coords[Rank]) const {
    Coordinate<Tensor1::Rank> c1(coords);
    uint8_t r1 = c1[d1];
    Coordinate<Tensor2::Rank> c2(&coords[Tensor1::Rank], d2, r1,
                                 &coords[Tensor1::Rank + d2]);
    ValueType value = t1_->Get(c1);
    // note that value will have already had the EffectivelyZero test.
    if (value == 0.)
      return 0;
    value *= t2_->Get(c2);
    if (EffectivelyZero(value))
      return 0;
    return value;
  }

  const ValueType Get(const Coordinate<Rank> &coords) const {
    return Get(coords.coords());
  }

  uint8_t Low(uint8_t d) const {
    assert(d < Rank);
    /* would be correct, but hurts performance
    if (d == d1)
      return std::max(t1_->Low(d1), t2_->Low(d2));
    else
    */
    if (d  < Tensor1::Rank)
      return t1_->Low(d);
    else if (d < Tensor1::Rank + d2)
      return t2_->Low(d - Tensor1::Rank);
    else
      return t2_->Low(d + 1 - Tensor1::Rank);
  }

  uint8_t High(uint8_t d) const {
    assert(d < Rank);
    /* would be correct, but hurts performance
    if (d == d1)
      return std::min(t1_->High(d1), t2_->High(d2));
    else
    */
    if (d  < Tensor1::Rank)
      return t1_->High(d);
    else if (d < Tensor1::Rank + d2)
      return t2_->High(d - Tensor1::Rank);
    else
      return t2_->High(d + 1 - Tensor1::Rank);
  }

  class Iterator {
   public:
    Iterator() {}
    Iterator(const ContractedTensor *t) : t_(t), i1_(t->t1_->begin()),
                                          i2_(t->t2_->begin()) {
      Next();
    }

    std::pair<Coordinate<Rank>, ValueType> &operator*() {
      SetValue();
      return val_;
    }

    std::pair<Coordinate<Rank>, ValueType> *operator->() {
      SetValue();
      return &val_;
    }

    // prefix ++
    Iterator &operator++() {
      Inc();
      Next();
      return *this;
    }

    void End(const ContractedTensor *t) {
      t_ = t;
      i1_ = t_->t1_->end();
      i2_ = t_->t2_->begin();
      set_ = false;
    }

    bool operator !=(const Iterator &other) const {
      return i1_ != other.i1_ || i2_ != other.i2_;
    }

    bool operator ==(const Iterator &other) const {
      return !(*this != other);
    }

   private:
    void Inc() {
      if (++i2_ == t_->t2_->end()) {
        i2_ = t_->t2_->begin();
        ++i1_;
      }
    }
    void Next() {
      set_ = false;
      while(i1_ != t_->t1_->end()
            && (i1_->first[d1] != i2_->first[d2]
                || EffectivelyZero(val_.second = i1_->second * i2_->second)))
        Inc();
    }

    void SetValue() {
      if (set_)
        return;
      set_ = true;

      assert(i1_->first[d1] == i2_->first[d2]);

      val_.first.Set(i1_->first, i2_->first.except(d2));
      // val_.second is already set in Next().
    }

    const ContractedTensor *t_;
    typename Tensor1::Iterator i1_;
    typename Tensor2::Iterator i2_;
    std::pair<Coordinate<Rank>, ValueType> val_;
    bool set_;
  };

  Iterator begin() const {
    return Iterator(this);
  }

  Iterator end() const {
    Iterator i;
    i.End(this);
    return i;
  }

  void Print(std::ostream &os) const {
    for (Iterator i(begin()); i != end(); ++i) {
      os << '[' << i->first << ": " << i->second << ']';
    }
  }

 private:
  const Tensor1 *t1_;
  const Tensor2 *t2_;
};

template <class Tensor1, rank_t d1, class Tensor2, rank_t d2>
std::ostream &operator<<(std::ostream &out,
                         const ContractedTensor<Tensor1, d1, Tensor2, d2> &t) {
  t.Print(out);
  return out;
}

template <class Tensor1, rank_t d1, rank_t d2> class SelfContractedTensor {
 public:
  static const rank_t Rank = Tensor1::Rank - 1;
  typedef typename Tensor1::ValueType ValueType;

  SelfContractedTensor(const Tensor1 *t)
      : t_(t) {
    assert(d1 < d2);
  }

  const ValueType Get(const uint8_t coords[Rank]) const {
    Coordinate<Tensor1::Rank> c(coords, d2, coords[d1], &coords[d2]);
    return t_->Get(c);
  }

  const ValueType Get(const Coordinate<Rank> &coords) const {
    return Get(coords.coords());
  }

  uint8_t Low(uint8_t d) const {
    assert(d < Rank);
    if (d < d2)
      return t_->Low(d);
    else
      return t_->Low(d + 1);
  }

  uint8_t High(uint8_t d) const {
    assert(d < Rank);
    if (d < d2)
      return t_->High(d);
    else
      return t_->High(d + 1);
  }

  class Iterator {
   public:
    Iterator() {}
    Iterator(const SelfContractedTensor *t) : t_(t), i_(t->t_->begin()) {
      Next();
    }

    std::pair<Coordinate<Rank>, ValueType> &operator*() {
      SetValue();
      return val_;
    }

    std::pair<Coordinate<Rank>, ValueType> *operator->() {
      SetValue();
      return &val_;
    }

    // prefix ++
    Iterator &operator++() {
      ++i_;
      Next();
      return *this;
    }

    void End(const SelfContractedTensor *t) {
      t_ = t;
      i_ = t_->t_->end();
      set_ = false;
    }

    bool operator !=(const Iterator &other) const {
      return i_ != other.i_;
    }

    bool operator ==(const Iterator &other) const {
      return i_ == other.i_;
    }

   private:
    void Next() {
      set_ = false;
      while(i_ != t_->t_->end()
            && i_->first[d1] != i_->first[d2])
        ++i_;
    }

    void SetValue() {
      if (set_)
        return;
      set_ = true;

      assert(i_->first[d1] == i_->first[d2]);
      val_.first = i_->first.except(d2);
      val_.second = i_->second;
    }

    const SelfContractedTensor *t_;
    typename Tensor1::Iterator i_;
    std::pair<Coordinate<Rank>, ValueType> val_;
    bool set_;
  };

  Iterator begin() const {
    return Iterator(this);
  }

  Iterator end() const {
    Iterator i;
    i.End(this);
    return i;
  }

  void Print(std::ostream &os) const {
    for (Iterator i(begin()); i != end(); ++i) {
      os << '[' << i->first << ": " << i->second << ']';
    }
  }

 private:
  const Tensor1 *t_;
};

template <class Tensor1, rank_t d1, rank_t d2>
std::ostream &operator<<(std::ostream &out,
                         const SelfContractedTensor<Tensor1, d1, d2> &t) {
  t.Print(out);
  return out;
}

// Note that for very sparse tensors the cache actually slows it
// down. Also, of course, the larger the tensor the larger the cache.
// Only seems to help for the very last tensor.
template <class Tensor1, rank_t d1, rank_t d2, bool cache = false>
class SelfContract2edTensor {
 public:
  static const rank_t Rank = Tensor1::Rank - 2;
  typedef typename Tensor1::ValueType ValueType;

  SelfContract2edTensor(const Tensor1 *t) : t_(t) {
    assert(d1 < d2);
  }

  void InnerCoord(Coordinate<Tensor1::Rank> *c, const uint8_t coords[Rank],
                  uint8_t x) const {
    for (rank_t to = 0, from = 0; to < Tensor1::Rank; ++to)
      if (to == d1)
        c->Set(to, x);
      else if (to == d2)
        c->Set(to, x);
      else
        c->Set(to, coords[from++]);
  } 

  const ValueType Get(const uint8_t coords[Rank]) const {
    ValueType ret = 0;
    uint8_t low = std::max(t_->Low(d1), t_->Low(d2));
    uint8_t high = std::min(t_->High(d1), t_->High(d2));
    for (uint8_t x = low; x <= high; ++x) {
      Coordinate<Tensor1::Rank> c;
      InnerCoord(&c, coords, x);
      ret += t_->Get(c);
    }
    if (EffectivelyZero(ret))
      return 0;
    return ret;
  }

  const ValueType Get(const Coordinate<Rank> &coords) const {
    return Get(coords.coords());
  }

  uint8_t Low(uint8_t d) const {
    assert(d < Rank);
    if (d < d1)
      return t_->Low(d);
    else if (d + 1 < d2)
      return t_->Low(d + 1);
    else
      return t_->Low(d + 2);
  }

  uint8_t High(uint8_t d) const {
    assert(d < Rank);
    if (d < d1)
      return t_->High(d);
    else if (d + 1 < d2)
      return t_->High(d + 1);
    else
      return t_->High(d + 2);
  }

  class Iterator {
   public:
    Iterator() {}
    Iterator(const SelfContract2edTensor *t) : t_(t), i_(t->t_->begin()) {
      Next();
    }

    const std::pair<Coordinate<Rank>, ValueType> &operator*() {
      SetValue();
      return val_;
    }

    const std::pair<Coordinate<Rank>, ValueType> *operator->() {
      SetValue();
      return &val_;
    }

    // prefix ++
    Iterator &operator++() {
      ++i_;
      Next();
      return *this;
    }

    void End(const SelfContract2edTensor *t) {
      t_ = t;
      i_ = t_->t_->end();
      set_ = false;
    }

    bool operator !=(const Iterator &other) const {
      return i_ != other.i_;
    }

    bool operator ==(const Iterator &other) const {
      return i_ == other.i_;
    }

   private:
    void Next() {
      set_ = false;
      typename Tensor1::Iterator end = t_->t_->end();
      for ( ; i_ != end; ++i_) {
        for ( ; i_ != end ; ++i_) {
          if (i_->first[d1] == i_->first[d2]) {
            if (!cache)
              break;
            Coordinate<Rank> coord = i_->first.except2(d1, d2);
            if (i_->second != 0. && done_.find(coord) == done_.end())
              done_.insert(coord);
              break;
          }
        }
        if (i_ == end || cache)
          break;
        if (i_->second != 0.) {
          Coordinate<Tensor1::Rank> new_coord = i_->first;
          uint8_t low = std::max(t_->t_->Low(d1), t_->t_->Low(d2));
          // Check whether we already handled or will handle this for
          // a lower value of the coordinate.
          for (uint8_t x = low; x < i_->first[d1] ; ++x) {
            new_coord.Set(d1, x);
            new_coord.Set(d2, x);
            if (t_->t_->Get(new_coord) != 0.)
              goto again;
          }
          break;
        again:
          continue;
        }
      }
    }

    void SetValue() {
      if (set_)
        return;
      set_ = true;

      assert(i_->first[d1] == i_->first[d2]);
      val_.first = i_->first.except2(d1, d2);
      Coordinate<Tensor1::Rank> new_coord = i_->first;
      val_.second = 0;
      uint8_t high = std::min(t_->t_->High(d1), t_->t_->High(d2));
      // Note that the value for x < i_->first[t_->d1_] is known to be
      // zero if we are not caching.
      for (uint8_t x = cache ? 0 : i_->first[d1]; x <= high ; ++x) {
        new_coord.Set(d1, x);
        new_coord.Set(d2, x);
        ValueType v = t_->t_->Get(new_coord);
        val_.second += v;
      }
      if (EffectivelyZero(val_.second))
        val_.second = 0;
    }

    const SelfContract2edTensor *t_;
    typename Tensor1::Iterator i_;
    std::pair<Coordinate<Rank>, ValueType> val_;
    bool set_;
    std::set<Coordinate<Rank> > done_;
  };

  Iterator begin() const {
    return Iterator(this);
  }

  Iterator end() const {
    Iterator i;
    i.End(this);
    return i;
  }

  void Print(std::ostream &os) const {
    for (Iterator i(begin()); i != end(); ++i) {
      os << '[' << i->first << ": " << i->second << ']';
    }
  }

 private:
  const Tensor1 *t_;
};

template <class Tensor1, rank_t d1, rank_t d2>
std::ostream &operator<<(std::ostream &out,
                         const SelfContract2edTensor<Tensor1, d1, d2> &t) {
  t.Print(out);
  return out;
}

template <class Tensor1, class Tensor2>
bool EqualTensors(const Tensor1 &t1, const Tensor2 &t2) {
  for (typename Tensor1::Iterator i = t1.begin(); i != t1.end(); ++i)
    if (i->second != t2.Get(i->first)) {
      std::cout << "one: " << i->first << ' ' << i->second << " != "
                << t2.Get(i->first) << std::endl;
      return false;
    }
  typename Tensor2::Iterator o;
  for (o = t2.begin(); o != t2.end(); ++o)
    if (o->second != t1.Get(o->first)) {
      std::cout << "two: " << o->first << ' ' << o->second << " != "
                << t1.Get(o->first) << std::endl;
      return false;
    }
  return true;
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
    uint8_t r1 = i1->first.coord(d1);
    std::pair<typename Map::const_iterator, typename Map::const_iterator> r2
        = t2map.equal_range(r1);
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

template <class OutTensor, class InTensor>
void ContractSelf(OutTensor *t_out, const InTensor &t_in,
                  uint8_t d1, uint8_t d2) {
  uint8_t d;
  if (d1 < d2)
    d = d2;
  else
    d = d1;

  typename InTensor::Iterator i1;
  for (i1 = t_in.begin(); i1 != t_in.end(); ++i1) {
    if (i1->first.coord(d1) == i1->first.coord(d2)) {
      Coordinate<InTensor::Rank - 1> new_coord;
      new_coord.Set(0, i1->first.except(d));
      t_out->Set(new_coord, i1->second);
    }
  }
}

void Interval() {
  static struct rusage prev;
  struct rusage cur;

  assert(getrusage(RUSAGE_SELF, &cur) == 0);
  struct timeval t;
  timersub(&cur.ru_utime, &prev.ru_utime, &t);
  double tt = t.tv_sec + t.tv_usec/1000000.;
  std::cout << "utime = " << tt << std::endl;

  prev = cur;
}

template <class OutTensor, class InTensor>
void ContractSelf2(OutTensor *t_out,
                   const InTensor &t_in, const uint8_t d1, const uint8_t d2,
                   const unsigned interval = 100000) {
  unsigned n = 0;
  typename InTensor::Iterator i1;
  for (i1 = t_in.begin(); i1 != t_in.end(); ++i1) {
    if (i1->first.coord(d1) == i1->first.coord(d2)) {
      if (++n == interval) {
        n = 0;
        std::cout << i1->first << ' ';
        Interval();
      }
      Coordinate<InTensor::Rank - 2> new_coord;
      new_coord.Set(0, i1->first.except2(d1, d2));
      t_out->Set(new_coord, t_out->Get(new_coord) + i1->second);
    }
  }
}

template <rank_t rank, class Value>
void Rearrange(Tensor<rank, Value> *t_out,
               const Tensor<rank, Value> &t_in, rank_t mapping[rank]) {
  typename std::map<Coordinate<rank>, Value>::const_iterator i;
  for (i = t_in.elements().begin(); i != t_in.elements().end(); ++i) {
    Coordinate<rank> new_coord;
    for (uint8_t j = 0; j < rank; ++j)
      new_coord.Set(j, i->first.coord(mapping[j]));
    t_out->Set(new_coord, t_in.Get(i->first));
  }
}

#if 0
template <rank_t rank0, rank_t rank1, rank_t rank2, rank_t rank3,
rank_t rank4, rank_t rank5, rank_t rank6, class Value>
void Contract6Tensors(Tensor<rank0, Value> *t_out,
    const Tensor<rank1, Value> &t1, rank_t d1[rank1],
    const Tensor<rank2, Value> &t2, rank_t d2[rank2],
    const Tensor<rank3, Value> &t3, rank_t d3[rank3],
    const Tensor<rank4, Value> &t4, rank_t d4[rank4],
    const Tensor<rank5, Value> &t5, rank_t d5[rank5],
    const Tensor<rank6, Value> &t6, rank_t d6[rank6]) {
  typedef std::multimap<rank_t, std::pair<rank_t, rank_t> *> Map;
  rank_t d;
  Map tmap;

#define M(x) \
  for (rank_t i##x = 0; i##x < rank##x; ++i##x) { \
    d = d##x[i##x]; \
    if (d < 0) \
      tmap.insert(std::pair<rank_t, \
          const std::pair<rank_t, rank_t> *>(d, (x, i##x))); \
  }

  M(2);
  M(3);
  M(4);
  M(5);
  M(6);

  for (rank_t i1 = 0; i1 < rank1; ++i1) {
    d = d1[i1];
    if (d < 0) {
      //find matching d in all other dx
      std::pair<typename Map::const_iterator,
                typename Map::const_iterator> r = tmap.equal_range(d);
      if (r.first != r.second) {
        typename Map::const_iterator j = r.first;
        uint8_t n = j->second->first;

#define T(x) \
        rank_t i##x = j->second->second; \
        tensor<rank1 + rank##x - 1> t; \
        Contract(&t, t1, i1, t##x, i##x);
        //wrong? only do this contraction the first time.



        for (++j; j != r.second; ++j) {

        }
      }
      else {

      }

      //contract over them one at a time
      //place in position d
      //set them all to 0 and remove the map entry
    }
    else if (d > 0) {
      //make free in that position
    }
  }
}
#endif

#include "auto_tensor.h"

