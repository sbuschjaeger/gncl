#!/usr/bin/env python3

# Modified version of https://github.com/kbruegge/cnn_cherenkov/blob/master/convert.py

import numpy as np
import photon_stream as ps
from tqdm import tqdm
import h5py
from fact.io import initialize_h5py, append_to_h5py
from astropy.table import Table
import click
import pickle
from joblib import Parallel, delayed
from itertools import islice


def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))


mapping = pickle.load(open('./hexagonal_position_dict.p', 'rb'))
mapping_fact_tools = np.array([t[1] for t in mapping.items()])

def remap_pixel_values(photon_content):
    input_matrix = np.zeros([46, 45])
    for i in range(1440):
        x, y = mapping_fact_tools[i]
        input_matrix[int(x)][int(y)] = photon_content[i]
    return input_matrix


def image_from_event(event, roi=[5, 40]):
    # print("IMAGE_FROm_EVENT")
    # d = {}
    # try:
    #     d['night'] = int(event.observation_info.night)
    #     d['run'] = int(event.observation_info.run)
    #     d['event_num'] = int(event.observation_info.event)

    # except AttributeError:
    #     truth = event.simulation_truth
    #     d['reuse'] = int(truth.reuse)
    #     d['run'] = int(truth.run)
    #     d['event_num'] = int(truth.event)
    #     d['energy'] = truth.air_shower.energy
    #     d['impact_x'] = truth.air_shower.impact_x(truth.reuse)
    #     d['impact_y'] = truth.air_shower.impact_y(truth.reuse)
    #     d['corsika_phi'] = truth.air_shower.phi
    #     d['corsika_theta'] = truth.air_shower.theta

    # d['az'] = event.az
    # d['zd'] = event.zd

    sequence = event.photon_stream.image_sequence
    img = remap_pixel_values(sequence[roi[0]:roi[1]].sum(axis=0))
    return img

def write_to_hdf(recarray, filename, key='events'):
    with h5py.File(filename, mode='a') as f:
        if key not in f:
            initialize_h5py(f, recarray, key=key)
        append_to_h5py(f, recarray, key=key)

def convert_file(path):
    # print('Analyzing {}'.format(path))
    reader = ps.EventListReader(path)
    rows = np.array([image_from_event(event) for event in reader])
    return rows
    # print(rows.shape)
    # asdf
    # return Table(rows).as_array()
    # try:
    # try:
    #     reader = ps.SimulationReader(
    #         photon_stream_path=path,
    #     )
    # except Exception as e:
    #     print("ASDF")
    #     print(e)
    #     reader = ps.EventListReader(path)
    # try:
    #     print("CALLING STUFF")
    #     rows = [image_from_event(event) for event in reader]
    #     return Table(rows).as_array()
    # except ValueError as e:
    #     print('Failed to read data from file: {}'.format(path))
    #     print('PhotonStreamError: {}'.format(e))

@click.command()
@click.argument('in_files', nargs=-1)
@click.argument('out_name')
@click.option('--n_jobs', '-n', default=16)
@click.option('--n_chunks', '-c', default=256)
def main(in_files, out_name, n_jobs, n_chunks):
    '''
    Reads all photon_stream files and converts them to images.
    '''
    # import os
    # if os.path.exists(out_file):
    #     if not yes:
    #         click.confirm('Do you want to overwrite existing file?', abort=True)
    #     os.remove(out_file)
    files = sorted(in_files)

    # files = files[:50]

    file_chunks = np.array_split(files, n_chunks)
    data = []
    for i, fs in enumerate(tqdm(file_chunks)):
        try:
            if n_jobs > 1:
                results = Parallel(n_jobs=n_jobs, verbose=False, backend='multiprocessing')(map(delayed(convert_file), fs))
            else:
                results = list(map(convert_file, fs))

            data.extend(results)
        except Exception as e:
            print(e)

    data = np.concatenate(data,axis=0)
    # if X is None:
    #     X = results 
    # else:
    #     print(X.shape)
    #     print(results.shape)
    #     X = np.vstack((X,results))
    
    # print(X.shape)
    np.save(out_name, data)

    # print()
    #     for r in results:
    #         try:
    #             write_to_hdf(r, out_file)
    #         except:
    #             pass

    #     print('Successfully read {} files'.format(len(results)))

    # print('writing corsika headers to file')
    # for f in tqdm(files):
    #     reader = ps.SimulationReader(
    #         photon_stream_path=f,
    #     )
    #     thrown_events = Table(reader.thrown_events()).as_array()
    #     write_to_hdf(thrown_events, out_file, key='showers')


if __name__ == '__main__':
    main()