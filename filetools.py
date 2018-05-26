# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 23:02:51 2014

@author: worker
"""

import os
import shutil
from   utils import Base #sysrun
# import simplejson
import errno
import subprocess
import glob
# pip install scandir
import scandir

def get_line(file,substr):
    with open(file, 'r') as f:
        for line in f.readlines():
            if substr in line:
                return line

def updirs(path,n):
    ''' Return number of directories up '''
    return os.path.abspath(os.path.join(os.path.dirname(path)+'/', '../'*n))

def list_dirtree(root,ext):
    '''List directory tree for files with matching extension '''
    fnames = []
    for root, dirs, files in scandir.walk(root,followlinks=True):
        for file in files:
            if file.find(ext) > -1:
                #fnames.update({file:root}) #append(os.path.join(root,file))
                fnames.append(os.path.join(root,file))
    return fnames

def names2cmd(names):
    cmd = 'printf \"'
    for name in names:
        cmd += (name + '\n')
    return cmd[:-1]

def replace_ext(path, new_ext):
   root, ext = os.path.splitext(path)
   return root + new_ext
   
def mkdir_assure(path):
    if os.path.exists(path):
        return 
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise  # raises the error again     

def rm_file_list(files):
    for f in files:
        os.remove(f)    
        
def rmtree_assure(path):
    if os.path.exists(path):
        shutil.rmtree(path)  
        
def rmtree_delayed(path): 
    if not os.path.exists(path):
        return None
    # rename the directory
    path  = os.path.normpath(path)
    bname = os.path.basename(path)
    npath = os.path.join(os.path.dirname(path),bname+'_deleting')
    # just in case of leftovers
    #rmtree_assure(npath)   
    if os.path.exists(npath):       
        subprocess.check_call(('rm', '-rf', npath))
    # rename initial directory
    subprocess.check_call(('mv', path, npath))
    # start process of removal
    p = subprocess.Popen(('rm', '-rf', npath))
    return p
    
def rm_assure(fname):
    if os.path.exists(fname):
        os.remove(fname)            

def file_only(path):
    return replace_ext(os.path.basename(path),'') 
    
def filenames(fullnames, ext):
    ''' return filename wothout dir and extension '''
    return [os.path.basename(f).replace(ext, '') for f in fullnames]
            
def dif_dirs(src_dirs,src_ext,dst_dir,dst_ext):
    ''' Return difference between names in src_dir and dst_dir '''
    srcfiles = []
    dstfiles = []
    for src_dir in src_dirs:
        srcfiles = srcfiles + glob.glob(os.path.join(src_dir,'*'+src_ext))
        dstfiles = dstfiles + glob.glob(os.path.join(dst_dir,'*'+dst_ext))    
    srcfiles = filenames(srcfiles,src_ext) # [os.path.basename(f).replace(src_ext, '') for f in srcfiles]
    dstfiles = filenames(dstfiles,dst_ext) # [os.path.basename(f).replace(dst_ext, '') for f in dstfiles]    
    return list(set(srcfiles) - set(dstfiles))    
    
def last_dir(path):
    return os.path.basename(os.path.normpath(path))


####################### GARBAGE #########################

# def clean_dir(path):
#     #find /path/to/directory -type f -exec rm {} \;
#     sysrun("find %s -type f -exec rm -rf {} \; " % path, err_check=False)


# class SaveAble(Base):
#     '''Allows saving this object to a json file.
#        NOTE: all member variables shall be native python lists !!!'''
#     def __init__(self, *args, **kwargs):
#         super(SaveAble, self).__init__(*args, **kwargs)
#         self.__name = kwargs['base_name']
#     def fullname(self,base_dir):
#         return os.path.join(base_dir, self.__name)
#     def setname(self,name):
#         self.__name = name
#     def jsonname(self,base_dir):
#         return self.fullname(base_dir) + '.json'
#     def save_json(self,base_dir,fname=None):
#         if fname != None:
#             self.__name = fname
#         ''' Saves object in a human readable (json) form '''
#         with open(self.jsonname(base_dir), "w") as outfile:
#             simplejson.dump(self.__dict__, outfile, indent=4)
#     def load_json(self,base_dir):
#         with open(self.jsonname(base_dir), "r") as infile:
#             self.__dict__ = simplejson.load(infile)

# def list_dirtree(in_dir,ext):
#     filenames   = {}
#     for filename in os.listdir(in_dir):
#         path = os.path.join(in_dir, filename)
#         if os.path.isdir(path):
#             filenames.update(list_dirtree(path,ext))
#         else:
#             if filename.endswith(ext):
#                 filenames.update({filename:in_dir})
#     return filenames
# def remote_ls(path,ext='',min_size_mb=0):
#     DELIM = 'MYSYNCDELIM'
#     # determine if this is an scp path
#     spath = path.split(':')
#     if len(spath) == 1:
#         # this is a regular path
#         #cmd = 'echo %s; find %s -maxdepth 0 -size +%dM' % (DELIM,os.path.join(path,'*'+ext),min_size_mb)
#         cmd = 'echo %s; find %s -name %s -size +%dM' % (DELIM,path,'*'+ext,min_size_mb)
#     else:
#         spath[0] = spath[0].replace('-P','-p')
#         cmd      = 'ssh %s "echo %s; find %s -name %s -size +%dM"' % (spath[0],DELIM,spath[1],'*'+ext,min_size_mb)
#         #cmd      = 'ssh %s "echo %s; find %s/*%s -maxdepth 0 -size +%dM"' % (spath[0],DELIM,spath[1],ext,min_size_mb)
#     print cmd
#     out,err,status = sysrun(cmd, err_check=False)
#     if err.find('No such file or directory') == 0:
#         return []
#     lines   = out.split('\n')
#     # look for delimiter line
#     for k in range(len(lines)):
#         if lines[k].find(DELIM) == 0:
#             k += 1
#             break;
#     # remove empty lines at the end
#     for l in range(k,len(lines)):
#         if lines[l] == '':
#             break;
#     return lines[k:l]

# def remote_copy(srcfname, destdir):
#     # determine if this is an scp path
#     spath    = srcfname.split(':')
#     if len(spath) == 1:
#         # this is a regular file, copy it
#         cmd = "cp -Lf %s %s/" % (srcfname, destdir)
#         #Movie.__msg("Local copying", cmd)
#     else:
#         spath[0] = spath[0].replace('-p','-P')
#         cmd      = "scp -v " + spath[0] +  ":\"" + spath[1] + "\" " + destdir
#         #Movie.__msg("Remote copying", cmd)
#     out,err,status = sysrun(cmd)
#     assert(status)


