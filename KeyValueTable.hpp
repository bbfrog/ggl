
#ifndef GGL_KEYVALUETABLE_HPP
#define GGL_KEYVALUETABLE_HPP

#include <map>
#include <graphlab.hpp>

namespace ggl {

//Is the value corresponding to the key associated with self-node, other-node, or edge in the context
//NONE is a special case to indicate the value is not associated with any position,
//e.g. the parameter gradient of the same function from different positions;
//     and the output of the same cost function from different positions are lumpped/summed together
enum POSITION {SELF, OTHER, EDGE, NONE};

//Is the value corresponding to the key associated with the INPUT data, HIDDEN layer codes, or (inputs, outputs, gradients) of a FUNCTION
enum TTYPE {INPUT, HIDDEN, FUNCTION};

struct ValueKey {
  POSITION m_Position;

  TTYPE m_TType;

  // if (m_TType == INPUT) => (m_Idx is the data type index)
  // if (m_TType == HIDDEN) => (m_Idx is the Hidden Layer index)
  // if (m_TType == FUNCION) => (m_Idx is the function index)
  // => to unqiuely identify function outputs cross all layers, a function index has to map to only one unique layerID
  // => !!!!WE CANNOT SUPPORT FUNCTION SHARING CROSS LAYERS !!!!
  // However, cross layer parameter sharing is not well defined in our case anyway.
  // TODO: have code check the model has no cross layer function/parameter sharing at the model compilation/verification stage
  // TODO: reporting error to the user at the very eary stage to avoid suprises
      
  int m_Idx;

  //Default constructor required??
  ValueKey(): m_Position(SELF), m_TType(INPUT), m_Idx(-1) {};

  ValueKey(POSITION position, TTYPE TType, int idx): m_Position(position), m_TType(TType), m_Idx(idx) {}

  bool operator< (const ValueKey& r) const {
    if (m_Position != r.m_Position) {
      return (m_Position < r.m_Position);
    } 
    else if (m_TType != r.m_TType) {
      return (m_TType < r.m_TType);
    }
    else {
      return (m_Idx < r.m_Idx);
    }
  }

  bool operator()(const ValueKey& l, const ValueKey& r ) const {
    if (l.m_Position != r.m_Position) {
      return (l.m_Position < r.m_Position);
    } 
    else if (l.m_TType != r.m_TType) {
      return (l.m_TType < r.m_TType);
    }
    else {
      return (l.m_Idx < r.m_Idx);
    }
  }

  const bool operator==(const ValueKey& rhs) {
    return (rhs.m_Position==m_Position && rhs.m_TType==m_TType && rhs.m_Idx==m_Idx);
  }    

  //Since the same data at the SELF position from one vertex's perspective is also the OTHER position from the other (neighbor) vertex's perspective,
  //we may need to "flip" the perspective when lookup the data from the inputValueTable generated from the other vertex
  const ValueKey Flip() {
    return ValueKey(Flip(m_Position), m_TType, m_Idx);
  }

  const POSITION Flip(POSITION p) {
    return (SELF == p) ? OTHER : ((OTHER == p)? SELF : EDGE);
  }

  const bool IsInvolveEdge() const {
    return (m_Position == OTHER || m_Position == EDGE) ? true : false;
  }
};

  //late binding the Value decision, we want to keep the following options open: 
  //(1) float vs. double
  //(2) 1 dimensional vector vs. high dimension tensor
  //(3) we should be able to experiment with them later on

//Vector to storage Value pointers
template<class Value>
class ValuePointerVector {
  public:
    void push_back(Value* pv) {
      vec.push_back(pv);
    }

    Value& operator[] (int idx) {
      return *vec[idx];
    }

    size_t size() {
      return vec.size();
    }

  private:
    std::vector<Value*> vec;
};

//Table to storage ValueKey and responding Value pointer
template<class Value>
class ValuePointerTable {
  public:
    void push_back(const ValueKey& vk, Value& v) {
      //assert vk should not in m_ValueTable
      m_ValueTable[vk] = &v;
    }

    void copy_and_push_back(const ValueKey& vk, const Value& v) {
      //assert vk should not in m_ValueTable
      //use deque instead of vector because deque push_back function doesn't change the previous elements pointer/reference
      //while vector push_back may change
      m_dequeValue.push_back(v);
      push_back(vk, m_dequeValue.back());
    }

    Value* operator[](const ValueKey& vk) {
      return m_ValueTable[vk];
    }

  private:
    std::map<ValueKey, Value*> m_ValueTable;
    std::deque<Value> m_dequeValue;
};

template<class Value>
class ValueTable {
  private:    
    struct WeightValue{
      Value value;
      float weight;
      WeightValue(const Value& v, float& w): value(v), weight(w) {
      }
      WeightValue(){};
      WeightValue operator+=(const WeightValue& rhs) {
        weight += rhs.weight;
        //it is possible current value or rhs.value is empty
        if(value.size() == 0) {
          value = rhs.value;
        }
        else if(rhs.value.size() > 0) {
          //both value and rhs.value are not empty
          value += rhs.value;
        }
        //else: value.size() > 0 and rhs.value.size() == 0, do nothing
        return *this;
     }
   };

  public:

    //TODO: Need to implement this!!! required by graphLab
    ValueTable() { }

    void save(graphlab::oarchive& arc) const {
      arc << m_ValueTable.size();
      typename std::map<ValueKey, WeightValue>::const_iterator it;
      for(it = m_ValueTable.begin(); it != m_ValueTable.end(); ++ it) {
        const ValueKey& key = it->first;
        const WeightValue& v = it->second;
        arc << key.m_Position << key.m_TType << key.m_Idx;
        arc << v.weight << v.value;
      }
    }

    void load(graphlab::iarchive& arc) {
      size_t size;
      arc >> size;
      m_ValueTable.clear();
      for(size_t i = 0; i < size; ++ i) {
        ValueKey key;
        float weight;
        Value v;
        arc >> key.m_Position >> key.m_TType >> key.m_Idx;
        arc >> weight >> v;
        Insert(key, v, weight);
      }
    }

    ValueTable<Value> operator+=(const ValueTable<Value>& rhs) {
      typename std::map<ValueKey, WeightValue>::const_iterator it; 
      for (it = rhs.m_ValueTable.begin(); it != rhs.m_ValueTable.end(); ++it) {
        Insert(it->first, it->second.value, it->second.weight);
      }
      return *this;
    }

    void Insert(const ValueKey& key, const Value& value, float weight=1.0) {
      WeightValue wv(value, weight);
      if(m_ValueTable.count(key) == 0) {
        m_ValueTable[key] = wv;
      }
      else {
        m_ValueTable[key] += wv;
      }
    }

    void erase(const ValueKey& key) {
      m_ValueTable.erase(key);
    }

    Value& operator[](const ValueKey& vk) {
      return m_ValueTable[vk].value;
    }
    
    float Weight(const ValueKey& vk) {
      return m_ValueTable[vk].weight;
    }

    //for the purpose of travsel the m_ValueTable
    void GetAllKeys(std::vector<ValueKey>& result) const {
      typename std::map<ValueKey, WeightValue>::const_iterator it;
      for(it = m_ValueTable.begin(); it != m_ValueTable.end(); ++ it) {
        result.push_back(it->first);
      }
    }

    size_t Size() const { return m_ValueTable.size(); }

    bool KeyExist(const ValueKey& key) const { return m_ValueTable.count(key) != 0; }

    void CopyToPointerTable(ValuePointerTable<Value>& valuePointerTable) const {
      typename std::map<ValueKey, WeightValue>::const_iterator it;
      for(it = m_ValueTable.begin(); it != m_ValueTable.end(); ++ it) {
        valuePointerTable.copy_and_push_back(it->first, it->second.value);
      }
    }

    //erase all items with the position of the key == pos
    void EraseByPosition(POSITION pos) {
      std::vector<ValueKey> keysToErase;

      typename std::map<ValueKey, WeightValue>::const_iterator it;
      for(it = m_ValueTable.begin(); it != m_ValueTable.end(); ++ it) {
        if(it->first.m_Position == pos) {
          keysToErase.push_back(it->first);
        }
      }

      for(size_t i = 0; i < keysToErase.size(); ++ i) {
        m_ValueTable.erase(keysToErase[i]);
      }
    }

  private:
    std::map<ValueKey, WeightValue> m_ValueTable;
};


} //namspace ggl

#endif //GGL_KEYVALUETABLE_HPP
