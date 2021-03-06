import sys
#sys.path.append('./androguard')
#sys.path.append('/usr/lib/python3/dist-packages')
sys.path.append('/root/anaconda2/lib/python2.7/site-packages')
#/root/anaconda2/lib/python2.7/site-packages
from androguard.core.bytecodes import apk, dvm
from androguard.core.analysis import analysis
import numpy as np
import re
import cPickle
import os

max_h = 100
max_calls =100
      

def gen_dict(list_path):

	file = open("error_files.txt","w") 

	external_api_dict = {}

	for path1 in list_path:

		count = 0

		


		
		for f in os.listdir(path1):

			# print count

			count+=1

			if(count==300):
		   		break

			path = os.path.join(path1,f)
			print (path)
		  
			try:
				if path.endswith('.apk'):
					app = apk.APK(path)
					app_dex = dvm.DalvikVMFormat(app.get_dex())
				else: 
					app_dex = dvm.DalvikVMFormat(open(path, "rb").read())
				app_x = analysis.newVMAnalysis(app_dex)

				methods = []
				cs = [cc.get_name() for cc in app_dex.get_classes()]

				ctr = 0
			  # print len(app_dex.get_methods())
				for method in app_dex.get_methods():
				    g = app_x.get_method(method)


				    

				    if method.get_code() == None:
				      continue




				    for i in g.get_basic_blocks().get():





				      for ins in i.get_instructions():
				        # This is a string that contains methods, variables, or
				        # anything else.

				        output = ins.get_output()

				        match = re.search(r'(L[^;]*;)->[^\(]*\([^\)]*\).*', output)
				        if match and match.group(1) not in cs:
				          methods.append(match.group())
				          # print "instruction : ", ins.get_basic_blocks()

				          # print "external api detected: ", match.group()
				          if(not external_api_dict.has_key(match.group())):
				            external_api_dict[match.group()] = len(external_api_dict)
			except:

				file.write(path) 

	file.close()

	    	


	return external_api_dict


if __name__ == '__main__':

  common_dict = gen_dict(["Dataset/all_benign","Dataset/all_drebin"])

  fp = open('15IT201_15IT217_M1_common_dict_300' + '.save', 'wb')
  cPickle.dump(common_dict, fp, protocol = cPickle.HIGHEST_PROTOCOL)
  fp.close()

  print (len(common_dict))
