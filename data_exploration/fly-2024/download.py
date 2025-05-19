from dandi.dandiapi import DandiAPIClient
import os


dandiset_id = '000727'
subject_ids = ['087', '089', '094', '097', '098', '099', '100', '101', '105']

nwbfiles = []

with DandiAPIClient() as client:
    dandiset = client.get_dandiset(dandiset_id, '0.240106.0043')
    assets_nwb = [asset for asset in list(dandiset.get_assets()) if asset.path.endswith('.nwb')]

    for asset_nwb in assets_nwb:
        print(asset_nwb.path)
        if str(asset_nwb.path) != 'sub-fly-087/sub-fly-087_ses-20200228161226_behavior+image+ophys.nwb': continue
        print(f"Downloading: {asset_nwb.path}")
        os.makedirs(os.path.join('../data/fly-2024', os.path.dirname(asset_nwb.path)), exist_ok=True)
        asset_nwb.download(os.path.join('../data/fly-2024', asset_nwb.path))
