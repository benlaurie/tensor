#include <sys/types.h>

#include <iostream>
#include <map>
#include <set>

template <uint8_t rank> class Coordinate {
public:
  explicit Coordinate(const uint8_t coords[rank]) {
    memcpy(coords_, coords, sizeof coords_);
  }
  Coordinate() {}
  bool operator<(const Coordinate &other) const {
    for (uint8_t d = 0; d < rank; ++d)
      if (coords_[d] < other.coords_[d])
	return true;
      else if (coords_[d] > other.coords_[d])
	return false;
    return false;
  }
  void Print(std::ostream &os) const {
    for (uint8_t r = 0; r < rank; ++r) {
      if (r != 0)
	os << ", ";
      os << (int)coords_[r];
    }
  }
  template <uint8_t rank2> void Set(uint8_t offset, Coordinate<rank2> coords) {
    for (uint8_t r = 0; r < rank2; ++r)
      coords_[r + offset] = coords.coord(r);
  }
  void Set(uint8_t r, uint8_t coord) { coords_[r] = coord; }
      
  const uint8_t coord(uint8_t r) const { return coords_[r]; }

private:
  uint8_t coords_[rank];
};

template <uint8_t rank> std::ostream &operator<<(std::ostream &out, const Coordinate<rank> coord) {
  coord.Print(out);
  return out;
} 

template <uint8_t rank, class Value> class Element {
public:
  Element(const Coordinate<rank> &coord, Value value)
    : coord_(coord), value_(value) {}
  bool operator<(const Element &other) const {
    return coord_ < other.coord_;
  }
  void Print(std::ostream &os) const {
    os << '[' << coord_ << ": " << value_ << ']';
  }
  const uint8_t coord(uint8_t r) const { return coord_.coord(r); }
  const Coordinate<rank> &coord() const { return coord_; }
  const Coordinate<rank - 1> coordExcept(uint8_t d) const {
    Coordinate<rank - 1> result;
    for (uint8_t r = 0; r < rank; ++r)
      if (r < d)
	result.Set(r, coord_.coord(r));
      else if (r > d)
	result.Set(r - 1, coord_.coord(r));
    return result;
  }
  const Value &value() const { return value_; }
private:
  Coordinate<rank> coord_;
  Value value_;
};

template <uint8_t rank, class Value> std::ostream &operator<<(std::ostream &out, const Element<rank, Value> element) {
  element.Print(out);
  return out;
} 

template <uint8_t rank, class Value> class Tensor {
public:
  typedef std::set<Element<rank, Value> > TElement;

  void Set(uint8_t coords[rank], Value value) {
    Coordinate<rank> coord(coords);
    Set(coord, value);
  }
  void Set(Coordinate<rank> coord, Value value) {
    Element<rank, Value> e(coord, value);

    // FIXME: erase wouldn't be needed if this were a map.
    elements_.erase(e);
    elements_.insert(e);
  }
  const Value &Get(Coordinate<rank> coord) const {
    Element<rank, Value> e(coord, 0);
    typename std::set<Element<rank, Value> >::const_iterator i = elements_.find(e);
    if (i == elements_.end()) {
      static Value zero(0);
      return zero;
    }
    return i->value();
  }
  void Print(std::ostream &os) const {
    typename std::set<Element<rank, Value> >::const_iterator i;
    for (i = elements_.begin();
	 i != elements_.end(); ++i) {
      os << *i;
    }
  }
  const std::set<Element<rank, Value> > &elements() const { return elements_; }
private:
  // FIXME: make this a map<Coordinate<rank>, Value>?
  std::set<Element<rank, Value> > elements_;
};

template <uint8_t rank, class Value> std::ostream &operator<<(std::ostream &out, const Tensor<rank, Value> tensor) {
  tensor.Print(out);
  return out;
}

template <uint8_t rank1, uint8_t rank2, class Value>
Tensor<rank1 + rank2 - 1, Value>
Contract(const Tensor<rank1, Value> &t1, uint8_t d1,
	 const Tensor<rank2, Value> &t2, uint8_t d2) {
  typedef std::multimap<uint8_t, const Element<rank2, Value> *> Map;
  Map t2map;
  typename std::set<Element<rank2, Value> >::const_iterator i;
  for (i = t2.elements().begin(); i != t2.elements().end(); ++i)
    t2map.insert(std::pair<uint8_t, const Element<rank2, Value> *>(i->coord(d2), &*i));

  Tensor<rank1 + rank2 - 1, Value> result;
  typename std::set<Element<rank1, Value> >::const_iterator i1;
  for (i1 = t1.elements().begin(); i1 != t1.elements().end(); ++i1) {
    uint8_t r = i1->coord(d1);
    std::pair<typename Map::const_iterator, typename Map::const_iterator> r2 = t2map.equal_range(r);
    for (typename Map::const_iterator i2 = r2.first; i2 != r2.second; ++i2) {
      Coordinate<rank1 + rank2 - 1> new_coord;
      new_coord.Set(0, i1->coord());
      new_coord.Set(rank1, i2->second->coordExcept(d2));
      result.Set(new_coord, i1->value() * i2->second->value());
    }
  }
  return result;
}

template <uint8_t rank1, uint8_t rank2, class Value>
Tensor<rank1 + rank2 - 2, Value>
Contract2(const Tensor<rank1, Value> &t1, uint8_t d1,
	  const Tensor<rank2, Value> &t2, uint8_t d2) {
  typedef std::multimap<uint8_t, const Element<rank2, Value> *> Map;
  Map t2map;
  typename std::set<Element<rank2, Value> >::const_iterator i2;
  for (i2 = t2.elements().begin(); i2 != t2.elements().end(); ++i2)
    t2map.insert(std::pair<uint8_t, const Element<rank2, Value> *>(i2->coord(d2), &*i2));

  Tensor<rank1 + rank2 - 2, Value> result;
  typename std::set<Element<rank1, Value> >::const_iterator i1;
  for (i1 = t1.elements().begin(); i1 != t1.elements().end(); ++i1) {
    uint8_t r = i1->coord(d1);
    std::pair<typename Map::const_iterator, typename Map::const_iterator> r2 = t2map.equal_range(r);
    if (r2.first != r2.second) {
      for (typename Map::const_iterator i2 = r2.first; i2 != r2.second; ++i2) {
	Coordinate<rank1 + rank2 - 2> new_coord;
	new_coord.Set(0, i1->coordExcept(d1));
	new_coord.Set(rank1 - 1, i2->second->coordExcept(d2));
	result.Set(new_coord, result.Get(new_coord)
		   + i1->value() * i2->second->value());
      }
    }
  }
  return result;
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
};

int main(int argc, char **argv) {
  DTensor3 t1;

  t1.Set(1, 1, 2, 15);
  t1.Set(1, 1, 2, 23.1);
  t1.Set(1, 1, 3, 10);
  t1.Set(100, 234, 123, 42);

  Tensor<2, double> t2;
  uint8_t c3[] = { 2, 1 };
  t2.Set(c3, 1.5);
  uint8_t c4[] = { 2, 2 };
  t2.Set(c4, 2);
  uint8_t c6[] = { 3, 1 };
  t2.Set(c6, 2.5);

  Tensor<4, double> t3;
  t3 = Contract(t1, 2, t2, 0);

  Tensor<3, double> t4;
  t4 = Contract2(t1, 2, t2, 0);

  std::cout << t1 << std::endl;
  std::cout << t2 << std::endl;
  std::cout << t3 << std::endl;
  std::cout << t4 << std::endl;
}
