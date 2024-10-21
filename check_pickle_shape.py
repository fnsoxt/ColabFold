import pickle,gzip

msa = pickle.load(gzip.open("/home/xukui/jobs/dBjTmCebNJXfH6xjcqlg05LOAE1XMVwHGW4FSA/features.pkl.gz"))
for k in msa.keys(): print(k, msa[k].shape)
