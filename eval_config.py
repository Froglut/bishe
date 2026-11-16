from util import mkdir


# directory to store the results
results_dir = './results/'
mkdir(results_dir)

# root to the testsets
dataroot = './dataset/test/'

# list of synthesis algorithms
# vals = ['progan', 'stylegan', 'biggan', 'cyclegan', 'stargan', 'gaugan',
#         'crn', 'imle', 'seeingdark', 'san', 'deepfake', 'stylegan2', 'whichfaceisreal']

# # indicates if corresponding testset has multiple classes
# multiclass = [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]

# vals = ['seeingdark', 'stargan', 'whichfaceisreal']
# multiclass = [ 0, 0, 0]

vals = ['progan', 'stylegan', 'biggan', 'san', 'deepfake','crn', 'gaugan', 'imle']
multiclass = [1, 1, 0, 0, 0, 0, 0, 0]

# model
model_path = 'checkpoints/16dual_stream/model_epoch_best.pth'

