import json
import logging
import time

import matplotlib.pyplot as plt
import pandas as pd

start_time = time.time()

formatter = logging.Formatter('%(asctime)s : %(name)s :: %(levelname)s : %(message)s')
logger = logging.getLogger('main')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
console_handler.setLevel(logging.DEBUG)
logger.debug('started')

# read the input filename from a JSON file
settings_file = './settings.json'
logger.debug('settings file : %s' % settings_file)
with open(settings_file, 'r') as settings_fp:
    settings = json.load(settings_fp)

logger.debug('settings: %s' % settings)
# now let's load the big input file
input_folder = settings['input_folder']
logger.debug('input folder: %s' % input_folder)
input_file = settings['input_file']
logger.debug('input file: %s' % input_file)
full_input_file = input_folder + input_file
logger.debug('reading input data from %s' % full_input_file)

column_names = settings['input_columns']
logger.debug('we are using column names: %s' % column_names)
separator = settings['separator'] if 'separator' in settings.keys() else ','
data = pd.read_csv(full_input_file, header=None, names=column_names, sep=separator)
logger.debug(data.shape)
default_head = 5
logger.debug('data head: \n%s' % str(data.head(default_head)))
logger.debug('data describe: \n%s' % str(data.describe()))
logger.debug('correlations: \n%s' % str(data.corr()))

output_folder = settings['output_folder']

# draw the histogram(s)
data.hist()
histogram_output_file = settings['histogram_output_file']
full_histogram_output_file = output_folder + histogram_output_file
logger.debug('writing histogram to %s' % full_histogram_output_file)
plt.savefig(full_histogram_output_file)

# clear the current figure
plt.clf()

# plot correlation matrix
plt.imshow(data.corr())
plt.colorbar()
correlations_output_file = settings['correlations_output_file']
full_correlations_output_file = output_folder + correlations_output_file
logger.debug('writing correlations to %s' % full_correlations_output_file)
plt.savefig(full_correlations_output_file)


logger.debug('done')
finish_time = time.time()
elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
logger.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
