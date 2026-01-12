import nibabel as nib
import numpy as np
import os
import argparse
import subprocess
import shutil

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Brain extraction (skull-stripping) using mri_synthstrip.')

    parser.add_argument(
        '--mp2rage-inv2',
        type=str,
        required=False,
        help='Use given 2nd inversion recovery (proton density-weighted) image from an MP2Rage sequence to generate the brain mask.'
    )
    
    parser.add_argument(
        '--mp2rage-inv2x',
        type=str,
        required=False,
        help='Multiply the (unified) input volume by the given 2nd inversion recovery (proton density-weighted) image from an MP2Rage sequence to generate a denoised intermediate image to generate the brain mask from. See https://doi.org/10.1016/j.neuroimage.2013.12.012'
    )
    
    parser.add_argument(
        'input',
        type=str,
        help='Input volume (nifti)'
    )
    
    parser.add_argument(
        'output',
        type=str,
        help='Output volume (nifti) of skull-stripped brain. Brainmask is available in _mask.nii.gz'
    )

    args = parser.parse_args()
    assert not (args.mp2rage_inv2 and args.mp2rage_inv2x), 'Invalid arguments: Cannot specify both --mp2rage-inv2 and --mp2rage-inv2x'
    assert os.path.isfile(args.input), 'Error: Input file {} not found'.format(args.input)
    
    print('Brain extraction using mri_synthstrip ...')
    
    input_file = args.input
    output_file = args.output
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    if args.mp2rage_inv2:
        # use INV2 to generate brain mask
        input_file = args.mp2rage_inv2
        output_file = output_file[:-7] + '_INV2_noskull.nii.gz'
        print('Using {} to generate brain mask'.format(input_file))
    elif args.mp2rage_inv2x:
        # multiply UNI images by INV2 to generate a denoised image from which the brain mask is generated
        # as described in https://doi.org/10.1016/j.neuroimage.2013.12.012
        input_file = output_file[:-7] + '_UNIxINV2.nii.gz'
        src_nib = nib.load(args.input)
        input_data = src_nib.get_fdata(dtype=np.float32)
        inv2_data = nib.load(args.mp2rage_inv2x).get_fdata(dtype=np.float32)
        result = input_data * inv2_data
        nib.save(nib.Nifti1Image(result, src_nib.affine), input_file)
        output_file = output_file[:-7] + '_UNIxINV2_noskull.nii.gz'
        print('Using {} to generate brain mask'.format(input_file))
        
    
    # run mri_synthstrip
    # mri_synthstrip -i input -o output -m mask
    mask_file_tmp = output_file[:-7] + '_mask.nii.gz'
    cmd = ['mri_synthstrip', '-i', input_file, '-o', output_file, '-m', mask_file_tmp]
    print('Running: {}'.format(' '.join(cmd)))
    subprocess.check_call(cmd)
    
    if args.mp2rage_inv2 or args.mp2rage_inv2x:
        # rename mask from intermediate image
        mask_file = args.output[:-7] + '_mask.nii.gz'
        os.rename(mask_file_tmp, mask_file)
        
        # apply brain mask from intermediate image to original UNI image
        src_nib = nib.load(args.input)
        input_data = src_nib.get_fdata(dtype=np.float32)
        brainmask = nib.load(mask_file).get_fdata(dtype=np.float32)
        input_data[brainmask == 0] = 0
        nib.save(nib.Nifti1Image(input_data, src_nib.affine), args.output)
