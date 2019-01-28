import argparse
import hashlib
import os
import subprocess
import ssl
import urllib2
import zipfile
from os.path import join

import pandas as pd
from joblib import Parallel, delayed
from PIL import Image


DATASETS = {
    'herbarium_asteraceae_phenophase': (
        'https://zenodo.org/record/2548630/files/herbarium_asteraceae_phenophase_annotations.zip?download=1',
        '0f2e27deb7c47c6648335574f882d54c',
        'annotations.csv'
    ),
    'herbarium_fertility': (
        'https://zenodo.org/record/2548630/files/herbarium_fertility_annotations.zip?download=1',
        '85452b677be117451bdad72bd2c3a0f0',
        'metadata.csv'
    )
}

allowed_content_type = ['image/jpeg', 'image/tiff', 'application/octet-stream']


def _check_md5sum(md5sum, filename=None, data=None):
    if (data is None and filename is None) or (data and filename):
        raise Exception('exactly one of data or filename arguments must be different than None')

    h = hashlib.md5()

    if data:
        h.update(data)
    if filename:
        with open(filename, 'r') as f:
            h.update(f.read())

    return h.hexdigest() == md5sum


def download_annotations_files(dataset_path, url, md5sum):
    # Create directory if needed
    annotations_path = join(dataset_path, 'annotations')
    if not os.path.exists(annotations_path):
        os.makedirs(annotations_path)

    filename = join(annotations_path, 'annotations.zip')
    if not os.path.exists(filename) or not _check_md5sum(md5sum, filename):
        # Download file
        response = urllib2.urlopen(url, timeout=60)
        data = response.read()
        response.close()

        # Check md5sum
        if not _check_md5sum(md5sum, data=data):
            raise Exception('md5sum does not check for file {}'.format(filename))

        # Write zip file on disk
        with open(filename, 'wb') as f:
            f.write(data)

    # Unzip file
    with zipfile.ZipFile(filename, 'r') as myzip:
        myzip.extractall(annotations_path)


def _check_image_integrity(fp):
    try:
        img = Image.open(fp)

        if 'JPEG' not in img.format_description:
            raise ValueError(
                'Wrong format: {} instead of JPEG'
                ''.format(img.format_description)
            )

        img.load()
    except Exception as e:
        raise e
    else:
        return True


def _download_image_file(output_directory, index, url, check_integrity,
                         preprocess):
    print('{} {}'.format(index, url))
    filename = '{}.jpg'.format(index)
    temp_filename = '.' + filename
    file_path = os.path.join(output_directory, filename)
    temp_file_path = os.path.join(output_directory, temp_filename)

    # Jump image if already correctly downloaded
    if os.path.exists(file_path):
        return filename

    # Download file
    response = None
    try:
        # Open connection
        context = ssl._create_unverified_context()
        response = urllib2.urlopen(url, context=context, timeout=60)

        # Check that we download an image
        content_type = response.headers.getheader('Content-Type')
        if content_type not in allowed_content_type:
            raise ValueError('content-type={}'.format(content_type))

        # Download data
        data = response.read()

        # Check image integrity if asked
        if check_integrity:
            from cStringIO import StringIO
            file_jpgdata = StringIO(data)
            if not _check_image_integrity(file_jpgdata):
                raise Exception('image integrity issue')

        # Write file on disk
        with open(temp_file_path, 'wb') as f:
            f.write(data)

        # Perform preprocessing if asked
        if preprocess:
            subprocess.check_call([
                'convert', temp_file_path, '-resize', '600x900^',
                '-quality', '85', temp_file_path
            ])

        os.rename(temp_file_path, file_path)

    except Exception as e:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        if os.path.exists(file_path):
            os.remove(file_path)
        print('Error: {} : index={} url={}'.format(e, index, url))
        return None

    finally:
        if response is not None:
            response.close()

    return filename


def download_images(dataset_path, csv_filename, check_integrity,
                    preprocess, n_jobs):
    # Create directory if needed
    images_path = join(dataset_path, 'images')
    if not os.path.exists(images_path):
        os.makedirs(images_path)

    # Load CSV file
    csv_path = join(dataset_path, 'annotations', csv_filename)
    df = pd.read_csv(csv_path, index_col='id')

    # Download images file
    filenames = Parallel(n_jobs=n_jobs)(
        delayed(_download_image_file)(
            images_path, index, url, check_integrity, preprocess
        )
        for index, url in df['URL'].iteritems()
    )

    # Save images filenames to CSV
    df_filenames = pd.DataFrame(data={'image_filename': filenames}, index=df.index)
    filenames_csv_path = join(dataset_path, 'annotations', 'image_filenames.csv')
    df_filenames.to_csv(filenames_csv_path)

    print('Number of images that could not be downloaded: {}'.format(df_filenames.isnull().sum().values[0]))


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Downloads and formats herbarium phenology datasets',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--check_integrity', action='store_true',
        help='check integrity of images after download'
    )
    parser.add_argument(
        '--preprocess', action='store_true',
        help='preprocess the images by resizing them to 900x600 and setting JPEG quality to 85'
    )
    parser.add_argument(
        '--n_jobs', type=int, default=1,
        help='use several jobs to speed-up images download'
    )
    parser.add_argument(
        'dataset_name', choices=DATASETS.keys(),
        help='name of the dataset to download'
    )
    parser.add_argument(
        'datasets_path', type=str,
        help='where to save the datasets'
    )
    args = parser.parse_args()

    print('Processing {}...'.format(args.dataset_name))
    url, md5sum, csv_filename = DATASETS[args.dataset_name]
    dataset_path = join(args.datasets_path, args.dataset_name)

    print('Fetching annotations files...')
    download_annotations_files(dataset_path, url, md5sum)

    print('Downloading images...')
    download_images(dataset_path, csv_filename, args.check_integrity,
                    args.preprocess, args.n_jobs)

    print('Finished!')
