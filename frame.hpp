#ifndef GGL_FRAME_HPP
#define GGL_FRAME_HPP

#include <graphlab.hpp>
#include "function.hpp"
#include "CoreFunction/ActNormFunc.hpp"
#include "CoreFunction/MatMultFunc.hpp"
#include "data.hpp"
#include "util/eigen_serialization.hpp"

namespace ggl {

template<class Value>
class Stack {
  public:
    //TODO:: Stack construction during the model file parsing
    // Assume the total number of recursive layers is M
    // cTotalLayers includes: inputLayer (idx=0), recursiveLayers (1<=idx<=M), outputLayer (idx=M+1), costLayer (idx=M+2) == (M+3)
    // m_cHidden Layers includes: inputLayer (idx=0), recursiveLayers (1<idx<=M) == (M+1)
    Stack(int cTotalLayers, int idxStack, bool isVertexStack, bool isEmptyHiddensStack = false) : 
        m_StackIdx(idxStack), m_cHidden(cTotalLayers-2), m_VertexStack(isVertexStack), m_EmptyHiddensStack(isEmptyHiddensStack) {
      m_FuncLayerArray = new FunctionLayer<Value>[cTotalLayers];
      m_pDataLayer = new DataLayer<Value>;
      m_pActNormFuncArray = new ActNormFunc<Value>*[m_cHidden];
      for(int i = 0; i < m_cHidden; ++ i) {
        m_pActNormFuncArray[i] = NULL;
      }
    }

    ~Stack() {
      delete [] m_FuncLayerArray;
      delete m_pDataLayer;
      //Note: stack does not own the individual function objects which are owned by the model object
      //      hence, it only does the shallow delete
      delete [] m_pActNormFuncArray;
    }

    bool Load(std::map<int, std::string>& dataMap, std::map<int, Value>& dataBlob){
      return m_pDataLayer->Load(dataMap, dataBlob);
    }

    int StackIdx() {
      return m_StackIdx;
    }

    int cHidden() {
      return m_cHidden;
    } 

    bool IsVertexStack() { return m_VertexStack; }

    bool IsEmptyHiddensStack() { return m_EmptyHiddensStack; }

    bool ContainDataFunc(int dataFunIdx) const {
      return m_pDataLayer->ContainDataFunc(dataFunIdx);
    }
   
    int NumOfDataFunc() const {
      return m_pDataLayer->Size();
    }

    // 0<=idxLayer<=M+2: specify the functions for input+Recursive+output+cost layers
    void AddFunction(int idxLayer, Function<Value>* pFunc) {
      m_FuncLayerArray[idxLayer].AddFunction(pFunc);
    }

    // specify the active and normalization function for the idxLayer
    // Note: 0<=idxLayer<=M this is restricted to the input and recursive layer
    //       the active norm function for the output layer is PER OUTPUT-FUNCTION specific not layer specific !!!
    //       Hence for output layer (M+1), the active norm function specification in on the function itself
    void AddActNormFunction(int idxLayer, ActNormFunc<Value>* pFunc) {
      if(idxLayer < 0 || idxLayer >= m_cHidden) return;
      m_pActNormFuncArray[idxLayer] = pFunc; 
    }

    // add the input data functions
    void AddData(DataFunction<Value>* pDataFunc) {
      m_pDataLayer->AddFunction(pDataFunc);
    }

    //retrieve the function and data layers
    FunctionLayer<Value>& operator [](const int idxLayer) {
      return m_FuncLayerArray[idxLayer];
    }

    DataLayer<Value>* GetDataLayer() {
      return m_pDataLayer;
    }

    ActNormFunc<Value>* getActNormFunc(const int idxLayer) {
      if(idxLayer < 0 || idxLayer >= m_cHidden) return NULL; 
      return m_pActNormFuncArray[idxLayer]; 
    }

  private:    
    FunctionLayer<Value>* m_FuncLayerArray;
    DataLayer<Value>* m_pDataLayer;
    //m_pActNormFuncArray[i] corresponding to the Activation/Normalization function applied on the i-th hidden layer
    ActNormFunc<Value>** m_pActNormFuncArray;
    int m_StackIdx;
    int m_cHidden; //number of hidden layers: inputLayer + RecursiveLayers == (M+1)
    bool m_VertexStack;  //whether it is vertexStack or edgeStack
    bool m_EmptyHiddensStack;  //whether this stack has empty hiddens
};


template<class Value>
class Frame {
  public:

    //graphLab require the default constructor?? (being part of the vertex data??)
    //ideally, we want to move the default constructor into the private section: e.g. disable it
    //because the Frame (and its derivatives) should only be created by the virtual constructor Create.
    Frame() { }

    //make the destructor virtual
    virtual ~Frame() { }

    //disable other constructors
    //virtual constructors of the the Frame classes
    static Frame<Value>* Create(Stack<Value>* pStack, graphlab::iarchive& arc);

    static Frame<Value>* Create(Stack<Value>* pStack, std::map<int, std::string>& dataMap);

    virtual int StackIdx() {return -1;}

    virtual void save(graphlab::oarchive& arc) const {}

    virtual void PrepareInput(int idxLayer, POSITION position, ValuePointerTable<Value>& inputTable, bool isRequiredByCostLayer = false) {}

    virtual Value& HiddenGrad(int idxLayer) {static Value v; return v;}

    virtual void SetHidden(int idxLayer, Value& hiddenLayer) {}

    virtual void ResumeHidden(int idxLayer, Value& hiddenLayer) {}

    virtual bool HasHiddenValue() { return false; }

    virtual void SetHiddenGrad(int idxLayer, POSITION position, Value& gradPostNorm) {}

    virtual void BackwardActNorm(int idxLayer) {}
  
    virtual Value GetHideen(int idxLayer) {static Value v; return v;}

    virtual void UpdateGrad() {}

    virtual void ClearGrad() {}

    virtual void ComputeHiddenSS(int idxLayer, ValueTable<Value>& ssTable) {}

    virtual void ComputeHiddenGradSS(int idxLayer, ValueTable<Value>& ssTable) {}

    virtual void ComputeDataSS(ValueTable<Value>& ssTable) {}

    virtual void UpdateDataBySS() {}

    virtual void Forward(int idxLayer, ValuePointerTable<Value>& inputValueTable, POSITION framePos, POSITION outPos,  FuncEvalMark funcEvalMark,
                 ValueTable<Value>& outputTable, float weight=1.0) {}

    //all the functions in the layer have the same gradient, e.g. at the recursive lauer, each function has the same gradient
    //which is the gradient of the layer's hidden layer
    virtual void Backward(int idxLayer, Value& outputGrad, ValuePointerTable<Value>& inputTable, POSITION framePos, FuncEvalMark funcEvalMark,
                 ValueTable<Value>& gradTable, float weight=1.0) {}

    virtual void BackwardParam(int idxLayer, Value& outputGrad, ValuePointerTable<Value>& inputTable, POSITION framePos, FuncEvalMark funcEvalMark,
                 ValueTable<Value>& paramGradTable, float weight=1.0) {}

    //different functions in the layer have different gradient, e.g. at the output layer, each output function has its own gradient
    virtual void Backward(int idxLayer, ValueTable<Value>& outputGradTable, POSITION framePos, POSITION outPos, ValuePointerTable<Value>& inputTable, FuncEvalMark funcEvalMark,
                 ValueTable<Value>& gradTable, float weight=1.0) {}

    virtual void BackwardParam(int idxLayer, ValueTable<Value>& outputGradTable, POSITION framePos, POSITION outPos, ValuePointerTable<Value>& inputTable, FuncEvalMark funcEvalMark,
                 ValueTable<Value>& paramGradTable, float weight=1.0) {}

   //////////////////Delegate DataLayer operations////////////////////////////////
    virtual void ForwardData(POSITION position, ValuePointerTable<Value>& dataValueTable, bool isRequiredByCostLayer){}

    virtual void BackwardData(ValueTable<Value>& outputGradTable, POSITION position){}
                                                                                       
    virtual void BackwardParamData(ValueTable<Value>& outputGrad, POSITION position, ValueTable<Value>& paramGradTable) {}

private:

    static Frame* NullFrame() {
      Frame* s_FrameNull = new Frame();
      return s_FrameNull;
    }
};

template<class Value>
class FrameFull : public Frame<Value> {
  public:

    FrameFull(){}

    virtual ~FrameFull(){} 
    
    //the key in the dataMap is the index (e.g. name in the Model context) of the corresponding Data Function
    FrameFull(Stack<Value>* pStack, std::map<int, std::string>& dataMap) : m_evenTimeCall(true), m_pStack(pStack) {
      if(!m_pStack->Load(dataMap, m_DataBlob)) {
        //use LOG_FATAL to terminate the program here. Need to refactor the load logic outside of constructor function
        logstream(LOG_FATAL) << "\n Failed in loading data " << std::endl;
      }
      m_HiddenLayers.resize(m_pStack->cHidden());
    }

    FrameFull(Stack<Value>* pStack, graphlab::iarchive& arc) : m_pStack(pStack) {
      //(1) m_pStack initialized
        
      //(2) m_DataBlob 
      size_t size;
      arc >> size;
      int key;
      Value v;
      for(size_t i=0; i<size; i++) {
        arc >> key;
        arc >> v;
        m_DataBlob[key] = v;
      }
  
      //(3) m_HiddenLayers
      arc >> size;
      m_HiddenLayers.resize(size);
      for(size_t i=0; i<size; i++) {
        arc >> m_HiddenLayers[i];
      }
  
      //(4) m_curGrad and m_nextGrad
      arc >> m_curGrad;
      arc >> m_nextGrad;
      arc >> m_evenTimeCall;

      //(5) m_FuncData
      arc >> size;
      for(size_t i=0; i<size; i++) {
        arc >> key;
        arc >> v;
        m_FuncData[key] = v; 
      }
    }

    static void SetUseNextGradForEdge(bool useNextGradForEdge = false) {
      bool* pSetNextGradForEdge = SetNextGradForEdgeSingleton();
      *pSetNextGradForEdge = useNextGradForEdge;  
    }

    static bool UseNextGradForEdge() {
      bool* pSetNextGradForEdge = SetNextGradForEdgeSingleton();
      return *pSetNextGradForEdge;
    }

    virtual int StackIdx() {
      return m_pStack->StackIdx(); 
    }

    virtual void save(graphlab::oarchive& arc) const {
      //(1)idx of m_pStack is loaded before FrameFull (in VertexProgram.hpp)

      //(2) m_DataBlob
      arc << m_DataBlob.size();
      typename std::map<int, Value>::const_iterator it;
      for (it = m_DataBlob.begin(); it != m_DataBlob.end(); ++ it) {
        arc << it->first;
        arc << it->second;
      }
     
      //(3) m_HiddenLayers
      arc << m_HiddenLayers.size();
      for(size_t i=0; i < m_HiddenLayers.size(); ++i) {
        arc << m_HiddenLayers[i];
      }

      //(4) m_curGrad
      arc << m_curGrad;
      arc << m_nextGrad;
      arc << m_evenTimeCall;

      //(5) m_FuncData
      arc << m_FuncData.size();
      for (it = m_FuncData.begin(); it != m_FuncData.end(); ++ it) {
        arc << it->first;
        arc << it->second;
      }

    }
    
    virtual void PrepareInput(int idxLayer, POSITION position, ValuePointerTable<Value>& inputTable, bool isRequiredByCostLayer = false) {
      int idxInput = idxLayer-1;
 
      if (idxInput == -1) {
        //asking the InputFunction to prepare the input for the input layer 
        //(which is the InputFunction itself
        ForwardData(position, inputTable, isRequiredByCostLayer);    
      }
      else {
        ValueKey vk(position, HIDDEN, idxInput);
        ActNormFunc<Value>* pActNormFunc = m_pStack->getActNormFunc(idxInput);
        if(pActNormFunc != NULL && pActNormFunc->NeedNorm()) {
          //when Norm is need, m_HiddenLayers[idxInput] is the only the input of ActNormFunc, so need do forward here
          inputTable.copy_and_push_back(vk, pActNormFunc->Forward(m_HiddenLayers[idxInput]));
        }
        else {
          inputTable.push_back(vk, m_HiddenLayers[idxInput]);
        }
      }
       
      //get the needed Function Data.
      //Cost Layer may need the input but don't need the first layer's function data
      if(isRequiredByCostLayer && idxLayer == 0) return;
      FunctionLayer<Value>& funcLayer = (*m_pStack)[idxLayer];
      if(funcLayer.HasData()) {
        for(int i = 0; i < funcLayer.Size(); ++ i) {
          Function<Value>* func = funcLayer[i];
          if(func->HasData()) {
            int funcIdx = func->FuncIdx();
            ValueKey funcKey(position, FUNCTION, funcIdx);
            inputTable.push_back(funcKey, m_FuncData[funcIdx]);
          }
        }
      }
      
    } 

    virtual Value& HiddenGrad(int idxLayer) { return m_curGrad; }

    virtual void SetHidden(int idxLayer, Value& hiddenLayer) {
      //When there is norm function in this layer, keep the input of ActNormFunc as the hidden layer output as it is need for backward
      //otherwise, save the output of the activation function, because for current support activation functions (relu, sigmoid, tanh), the output of the function is need for back propagation
      ActNormFunc<Value>* pActNormFunc = m_pStack->getActNormFunc(idxLayer);
      if (pActNormFunc != NULL && !pActNormFunc->NeedNorm()) {
        m_HiddenLayers[idxLayer] = pActNormFunc->Forward(hiddenLayer);
      }
      else { 
        m_HiddenLayers[idxLayer] = hiddenLayer; 
      }
    }

    virtual void ResumeHidden(int idxLayer, Value& hiddenLayer) {
      m_HiddenLayers[idxLayer] = hiddenLayer;
    }

    virtual bool HasHiddenValue() {
      return m_HiddenLayers.size() > 0 && m_HiddenLayers[0].size() > 0;
    }

    virtual void SetHiddenGrad(int idxLayer, POSITION position, Value& gradPostNorm) {
      
      //Performance Hack ALERT: !!!
      //When there is Edge Stack, we need to use transform_edges to set edge gradient, then need to set m_nextGrad instead m_curGrad here
      //because GAS run after transform_edges and it still need m_curGrad of edges.
      bool setNextGrad = FrameFull<Value>::UseNextGradForEdge() && (position == EDGE);
      ActNormFunc<Value>* pActNormFunc = m_pStack->getActNormFunc(idxLayer);
      //if gradPostNorm.size() == 0, means no gradient passed down, doesn't need to call Activation function
      //Then ActNorm function doesn't need to check whethere there is a gradient passed down
      //If pActNormFunc need Norm, the ActNormFunc Backward need to be called after hidden Grad SS is calculated, so call backward here only when no need Norm
      if (NULL != pActNormFunc && gradPostNorm.size() > 0 && !pActNormFunc->NeedNorm()) {
        if(setNextGrad) {
          m_nextGrad = pActNormFunc->Backward(gradPostNorm, m_HiddenLayers[idxLayer]);
        }
        else {
          m_curGrad = pActNormFunc->Backward(gradPostNorm, m_HiddenLayers[idxLayer]);
        }
      }
      else {
        if(setNextGrad) {
          m_nextGrad = gradPostNorm;
        }
        else {
          m_curGrad = gradPostNorm;
        }
      }
    }

    virtual void BackwardActNorm(int idxLayer) {
      ActNormFunc<Value>* pActNormFunc = m_pStack->getActNormFunc(idxLayer);
      if(m_curGrad.size() == 0 || pActNormFunc == NULL || !pActNormFunc->NeedNorm()) return;

      m_curGrad = pActNormFunc->Backward(m_curGrad, m_HiddenLayers[idxLayer]);
    }

    virtual Value GetHideen(int idxLayer) {
      if(idxLayer < 0 || idxLayer >= (int)m_HiddenLayers.size()) {
        Value empty;
        return empty;
      }

      //Return the HiddenLayers Value directly, as this output is usually used for resuming
      ///NOTICE: If has norm function in this layer, this return value is before normalization.
      return m_HiddenLayers[idxLayer];
    }

    virtual void UpdateGrad() { m_curGrad = m_nextGrad; }

    virtual void ClearGrad() { m_curGrad.resize(0); }

    //compute the sufficent statistics required by the Normalization function
    virtual void ComputeHiddenSS(int idxLayer, ValueTable<Value>& ssTable) {
      ActNormFunc<Value>* pActNormFunc = m_pStack->getActNormFunc(idxLayer);
      if(pActNormFunc == NULL || !pActNormFunc->NeedNorm()) return;

      //use stack id as the index of the key
      ValueKey key(NONE, HIDDEN, m_pStack->StackIdx());
      Value ssValue = pActNormFunc->ComputeHiddenSS(m_HiddenLayers[idxLayer]);
      ssTable.Insert(key, ssValue);
    }

    virtual void ComputeHiddenGradSS(int idxLayer, ValueTable<Value>& ssTable) {
      ActNormFunc<Value>* pActNormFunc = m_pStack->getActNormFunc(idxLayer);
      if(m_curGrad.size() == 0 || pActNormFunc == NULL || !pActNormFunc->NeedNorm()) return;

      //use stack id as the index of the key
      ValueKey key(NONE, HIDDEN, m_pStack->StackIdx());
      Value ssValue = pActNormFunc->ComputeHiddenGradSS(m_curGrad, m_HiddenLayers[idxLayer]);
      ssTable.Insert(key, ssValue);
    }


    //compute the sufficent statistics required by the Data Layer function
    virtual void ComputeDataSS(ValueTable<Value>& ssTable) {
      m_pStack->GetDataLayer()->ComputeDataSS(m_DataBlob, ssTable);
    }

    virtual void UpdateDataBySS() {
      m_pStack->GetDataLayer()->UpdateDataBySS(m_DataBlob);
    }

   //////////////Delegate FunctionLayer operations/////////////////////////
    virtual void Forward(int idxLayer, ValuePointerTable<Value>& inputValueTable, POSITION framePos, POSITION outPos,  FuncEvalMark funcEvalMark, 
                 ValueTable<Value>& outputTable, float weight=1.0){
     (*m_pStack)[idxLayer].Forward(inputValueTable, framePos, outPos, funcEvalMark, outputTable, weight);
    }

    //all the functions in the layer have the same gradient, e.g. at the recursive lauer, each function has the same gradient
    //which is the gradient of the layer's hidden layer
    virtual void Backward(int idxLayer, Value& outputGrad, ValuePointerTable<Value>& inputTable, POSITION framePos, FuncEvalMark funcEvalMark, 
                 ValueTable<Value>& gradTable, float weight=1.0) {
      (*m_pStack)[idxLayer].Backward(outputGrad, inputTable, framePos, funcEvalMark, gradTable, weight);
    }
      
    virtual void BackwardParam(int idxLayer, Value& outputGrad, ValuePointerTable<Value>& inputTable, POSITION framePos, FuncEvalMark funcEvalMark, 
                 ValueTable<Value>& paramGradTable, float weight=1.0) {
      (*m_pStack)[idxLayer].BackwardParam(outputGrad, inputTable, framePos, funcEvalMark, paramGradTable, weight);
    }

    //different functions in the layer have different gradient, e.g. at the output layer, each output function has its own gradient
    virtual void Backward(int idxLayer, ValueTable<Value>& outputGradTable, POSITION framePos, POSITION outPos, ValuePointerTable<Value>& inputTable, FuncEvalMark funcEvalMark, 
                 ValueTable<Value>& gradTable, float weight=1.0) {
      (*m_pStack)[idxLayer].Backward(outputGradTable, framePos, outPos, inputTable, funcEvalMark, gradTable, weight);
    }

    virtual void BackwardParam(int idxLayer, ValueTable<Value>& outputGradTable, POSITION framePos, POSITION outPos, ValuePointerTable<Value>& inputTable, FuncEvalMark funcEvalMark, 
                 ValueTable<Value>& paramGradTable, float weight=1.0) {
      (*m_pStack)[idxLayer].BackwardParam(outputGradTable, framePos, outPos, inputTable, funcEvalMark, paramGradTable, weight);
    } 

   //////////////////Delegate DataLayer operations////////////////////////////////
    virtual void ForwardData(POSITION position, ValuePointerTable<Value>& dataValueTable, bool isRequiredByCostLayer){
      m_pStack->GetDataLayer()->Forward(m_DataBlob, position, dataValueTable, isRequiredByCostLayer);
    }

    virtual void BackwardData(ValueTable<Value>& outputGradTable, POSITION position){
      if(position == EDGE) {
        m_evenTimeCall = !m_evenTimeCall;
        if(!m_evenTimeCall) return;
      }
      m_pStack->GetDataLayer()->Backward(outputGradTable, position, m_DataBlob);
    }

    virtual void BackwardParamData(ValueTable<Value>& outputGrad, POSITION position, ValueTable<Value>& paramGradTable) {
      m_pStack->GetDataLayer()->BackwardParam(outputGrad, position, m_DataBlob, paramGradTable);
    }

  private:
    static bool* SetNextGradForEdgeSingleton() {
      static bool* s_pSetNextGradForEdge = new bool();
      return s_pSetNextGradForEdge;
    }
 
  private:
    std::vector<Value> m_HiddenLayers;
    std::map<int, Value> m_DataBlob;
    Value m_curGrad;
    //For save the next gradient and doesn't change the m_curGrad in case m_curGrad is still need
    //should only used for setting edge gradient
    //TODO: As in GAS, each edge will be visited twice, so can do the edge gradient set in the second time
    //Use m_evenTimeCall to indicate whether it is the second time call, and it is already used in backwardData
    //Idealy Recurisve layer and output layer should also use m_evenTimeCall instead of using m_nextGrad. Need to test in real data.
    Value m_nextGrad;
    bool m_evenTimeCall;

    //For save the funcData
    std::map<int, Value> m_FuncData;
    Stack<Value>* m_pStack;
};

//virtual constructors of the the Frame classes
template<class Value>
Frame<Value>* Frame<Value>::Create(Stack<Value>* pStack, graphlab::iarchive& arc) {
      if (NULL == pStack) {
        return NullFrame();
      }
      else {
        return new FrameFull<Value>(pStack, arc);
      }
}

//the key in the dataArray is the index (e.g. name in the context of the model) of the data function
template<class Value>
Frame<Value>* Frame<Value>::Create(Stack<Value>* pStack, std::map<int, std::string>& dataMap) {
      if (NULL == pStack) {
        return NullFrame();
      }
      else {
        return new FrameFull<Value>(pStack, dataMap);
      }
}

}

#endif //GGL_FRAME_HPP
