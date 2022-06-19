import glob,random

root='./'
lines=[]
paths1=glob.glob(root+'pngdata/style-ffhq/*.png')
for path in paths1:
  line=path+' 0\n'
  lines.append(line)

paths2=glob.glob(root+'pngdata/style-ffhq/*.png')
for path in paths2:
  line=path+' 1\n'
  lines.append(line)
  
f=open('list.txt', 'w')
random.shuffle(lines)
for line in lines:
  f.write(line)
f.close()


  