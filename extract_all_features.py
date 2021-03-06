import sys
#sys.path.append('/home/vikram_mm/.local/lib/python3.5/site-packages/')

from androguard.core.bytecodes import apk, dvm
from androguard.core.analysis import analysis
import numpy as np
import re
import cPickle
import os

max_h = 50
max_calls = 50


def extract_all_features():

  print "loading dict..."
  external_api_dict = cPickle.load( open( "15IT201_15IT217_M1_common_dict_300.save", "rb" ) )
  print "done!"


  path_list = ["Dataset/all_benign","Dataset/all_drebin"]
  
  index = 0
  for i in range(2):
	
    count = 0
    for path in os.listdir(path_list[i]):
      
      count+=1
     

      if(count==5):
	break

      index+=1

      print count,os.path.join(path_list[i],path)
      

      try:
      	#X.append(get_feature_vector(os.path.join(path_list[i],path), external_api_dict))
      	#Y.append(i)
	x = get_feature_vector(os.path.join(path_list[i],path), external_api_dict)
	data_point = {}
	data_point['x'] = x
	data_point['y'] = i
	#fp = open(os.path.join('features',str(index) + '.save'), 'wb')
	fp = open(os.path.join('15IT201_15IT217_M1_testcases_features',str(path) + '.save'), 'wb')
	cPickle.dump(data_point, fp, protocol = cPickle.HIGHEST_PROTOCOL)
	fp.close()	
      
      except:
	
	print "exception occured"
	#count=count-1


  #print X
  #print Y

  #print np.asarray(X)
  #print np.asarray(Y)

  #return np.asarray(X),np.asarray(Y)
  return

def get_feature_vector(path, external_api_dict):


  feature_vector = np.zeros((max_calls,len(external_api_dict),max_h),dtype=int)

  call_no = 0
  seq_no = 0

  if path.endswith('.apk'):
      app = apk.APK(path)
      app_dex = dvm.DalvikVMFormat(app.get_dex())
  else: 
      app_dex = dvm.DalvikVMFormat(open(path, "rb").read())

  app_x = analysis.newVMAnalysis(app_dex)


  cs = [cc.get_name() for cc in app_dex.get_classes()]


  # print len(app_dex.get_methods())
  for method in app_dex.get_methods():
    g = app_x.get_method(method)


    if method.get_code() == None:
      continue
  

    # print "***********"
    # print "method beeing investigated - ", g


    for i in g.get_basic_blocks().get():


      # print "i.childs : " ,i.childs

      if(i.childs!=[] and seq_no<max_h):

        call_no = 0
        for ins in i.get_instructions():

          output = ins.get_output()
          

          match = re.search(r'(L[^;]*;)->[^\(]*\([^\)]*\).*', output)
          if match and match.group(1) not in cs:
            
            # print "instruction : ", ins.get_basic_blocks()
            # print "output : ", output
            # print "external api detected: ", match.group()
            

            # if(i.childs!=[]):
              # print "-------->",i.childs[0][2].childs
              # break
            feature_vector[call_no,external_api_dict[match.group()],seq_no] = 1
            call_no+=1
     

        
        rand_child_selected = np.random.randint(len(i.childs))
        # print rand_child_selected

        traverse_graph(i.childs[rand_child_selected][2],feature_vector,cs,call_no,seq_no,external_api_dict)
        seq_no+=1

  return feature_vector


def traverse_graph(node,feature_vector,cs,call_no,seq_no,external_api_dict):


  for ins in node.get_instructions():


      output = ins.get_output()
      match = re.search(r'(L[^;]*;)->[^\(]*\([^\)]*\).*', output)

      if match and match.group(1) not in cs and call_no<max_calls:

        feature_vector[call_no,external_api_dict[match.group()],seq_no] = 1
        call_no+=1


  
  if(call_no<max_calls and node.childs!=[]):
  
    rand_child_selected = np.random.randint(len(node.childs))
    traverse_graph(node.childs[rand_child_selected][2],feature_vector,cs,call_no,seq_no,external_api_dict)


if __name__ == '__main__':

  #x,y = load_data()
  extract_all_features()


