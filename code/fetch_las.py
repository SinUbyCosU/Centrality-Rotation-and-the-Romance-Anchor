import paramiko, json, numpy as np

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('216.128.144.102', username='root', password='[8eE967Lg}!(GZoz')

_, out, _ = c.exec_command('cat /root/clr_paper/results/experiment14_las_validation.json')
data_str = out.read().decode()
if data_str.strip():
    d = json.loads(data_str)
    pre = []
    post = []
    for m in d:
        for l in d[m]:
            pre.append(d[m][l]['pre_las_sim'])
            post.append(d[m][l]['post_las_sim'])
    print(f"LAS Validation: Pre={np.mean(pre):.3f}, Post={np.mean(post):.3f} (n={len(pre)})")
else:
    print("LAS validation json is empty or missing")
