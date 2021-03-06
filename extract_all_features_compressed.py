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
  external_api_dict = cPickle.load( open( "common_dict_300.save", "rb" ) )
  print "done!"

  #X = []if __name__ if __name__ == '__main__':== '__main__':
  #Y = []

  #path_list = ["Dataset/benign","Dataset/all_drebin"]
  path_list = ["Dataset/all_benign"]
  index = 0
  for i in range(2):
	
    count = 0
    for path in os.listdir(path_list[i])[::-1]:
      
      count+=1
     

      if(count==34):
	break

      index+=1

      print count,os.path.join(path_list[i],path)
      

      try:
      	#X.append(get_feature_vector(os.path.join(path_list[i],path), external_api_dict))
      	#Y.append(i)
	x = get_compressed_feature_vector(os.path.join(path_list[i],path), external_api_dict)
	#print x.shape
        #print x
	#exit(0)
	data_point = {}
	data_point['x'] = x
	data_point['y'] = 1
	fp = open(os.path.join('features',str(index) + '.save'), 'wb')
	fp = open(os.path.join('all_compressed_features',str(path) + '.save'), 'wb')
	#fp = open(os.path.join('acf2',str(path) + '.save'), 'wb')
	cPickle.dump(data_point, fp, protocol = cPickle.HIGHEST_PROTOCOL)
	fp.close()	
      
      except Exception as e:
	
	print "exception occured"
	print e
	#count=count-1


  #print X
  #print Y

  #print np.asarray(X)
  #print np.asarray(Y)

  #return np.asarray(X),np.asarray(Y)
  return

def get_compressed_feature_vector(path, external_api_dict):


  feature_vector = np.zeros((max_calls,max_h),dtype=int)

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
          # This is a string that contains methods, variables, or
          # anything else.

          output = ins.get_output()
          

          match = re.search(r'(L[^;]*;)->[^\(]*\([^\)]*\).*', output)
          if match and match.group(1) not in cs and call_no<max_calls:
            
            # print "instruction : ", ins.get_basic_blocks()
            # print "output : ", output
            # print "external api detected: ", match.group()
            

            # if(i.childs!=[]):
              # print "-------->",i.childs[0][2].childs
              # break
            feature_vector[call_no,seq_no] = external_api_dict[match.group()]
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

        feature_vector[call_no,seq_no] = external_api_dict[match.group()]
        call_no+=1


  
  if(call_no<max_calls and node.childs!=[]):
  
    rand_child_selected = np.random.randint(len(node.childs))
    traverse_graph(node.childs[rand_child_selected][2],feature_vector,cs,call_no,seq_no,external_api_dict)

       

def main():
  """
  For test
  """

if __name__ == '__main__':

  #x,y = load_data()
  extract_all_features()

  

  '''print x.shape
  print y.shape

  np.save('x200.npy', x)
  np.save('y200.npy', y)'''



