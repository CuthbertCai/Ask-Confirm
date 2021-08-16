import sys
sys.path.append('../')
import tqdm
import threading

from datasets.vg import vg
from datasets.loader import region_loader
from utils.config import get_train_config
from utils.vocab import Vocabulary

same_obj = 0
idx = 0

def stat(loaddbs):
    objs = []
    obj_inds = []
    obj2count = {}
    no_obj_scene = 0
    one_obj_scene = 0
    for loaddb in loaddbs:
        for scene in loaddb:
            obj, obj_ind = scene[5], scene[6]
            if len(obj) == 0:
                no_obj_scene += 1
            elif len(obj) == 1:
                one_obj_scene += 1
            objs.append(obj)
            obj_inds.append(obj_ind)
            for o in obj:
                if o in obj2count:
                    obj2count[o] += 1
                else:
                    obj2count[o] = 1
    obj2count = sorted(obj2count.keys(), key= lambda kv: obj2count[kv])
    return objs, obj_inds, obj2count, no_obj_scene, one_obj_scene

def count_same_obj_scene(loaddbs):
    global same_obj
    global idx
    for loaddb in loaddbs:
        for scene in loaddb:
            idx += 1
            obj_ind = scene[6]
            print(obj_ind)
            for j in range(idx, len(loaddb)):
                obj_ind_j = loaddb[j][6]
                if set(obj_ind_j) == (obj_ind):
                    same_obj += 1
                    print('same_obj', same_obj)
    # return same_obj


def main():
    config, unparsed = get_train_config()
    loaddbs = []
    for split in ['train', 'test']:
        db = vg(config, split)
        loaddb = region_loader(db)
        loaddbs.append(loaddb)
    # objs, obj_inds, obj2count, no_obj_scene, one_obj_scene = stat(loaddbs)
    # print('no_obj_scene', no_obj_scene)
    # print('one_obj_scene', one_obj_scene)
    t = threading.Thread(target=count_same_obj_scene, args=(loaddbs,))
    t.start()
    t.join()
    # same_obj = count_same_obj_scene(loaddbs)
    # print('same obj scene', same_obj)
    # print('objs: ', objs)
    # print('obj_inds: ', obj_inds)
    # print('obj2count: ', obj2count)
    # top_objs = obj2count[-500:]
    # for i in range(len(objs)):
    #     count = 0
    #     for o in objs[i]:
    #         if o in top_objs:
    #             count += 1
    #     if count == 0:
    #         print(i)
    # file_path = '/data/home/cuthbertcai/programs/DiaVL/data/caches/vg_objects_vocab_500.txt'
    # with open(file_path, 'w') as f:
    #     for obj in top_objs:
    #         f.write(obj + '\n')
if __name__ == '__main__':
    main()



