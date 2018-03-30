mkdir .data
cd .data
wget https://static.aminer.org/lab-datasets/citation/dblp.v10.zip
sudo apt-get install unzip
unzip dblp.v10.zip
wget https://static.aminer.org/lab-datasets/citation/citation-acm-v8.txt.tgz
tar -xzf citation-acm-v8.txt.tgz
rm dblp.v10.zip citation-acm-v8.txt.tgz