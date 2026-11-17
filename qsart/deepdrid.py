"""Audit official DeepDRiD regular-fundus *metadata* CSVs (no images downloaded)."""
import argparse
import json
from pathlib import Path
import pandas as pd

EXPECTED = {'patient_id', 'image_id', 'Overall quality',
            'left_eye_DR_Level', 'right_eye_DR_Level', 'patient_DR_Level',
            'Clarity', 'Field definition', 'Artifact'}


def normalize_labels(path, split_name):
    """Map LEFT/RIGHT EYE labels, never the more severe patient_DR_Level.

    Dataset uses image_id examples 1_l1, 1_l2, 1_r1, 1_r2.
    Overall quality: original binary flag, not an inferred gradability score.
    """
    return normalize_frame(pd.read_csv(path), split_name, source=str(path))


def normalize_frame(df, split_name, source='dataframe'):
    """Normalize verified conventional rows; retain strict historical semantics.

    The manifest pipeline quarantines official filename/eye-column conflicts
    before invoking this function, rather than guessing corrected eye labels.
    """
    missing = EXPECTED - set(df.columns)
    if missing: raise ValueError(f'{source}: missing official columns {sorted(missing)}')
    side = df.image_id.astype(str).str.extract(r'_([lr])[12]$', expand=False)
    if side.isna().any(): raise ValueError('Unrecognized DeepDRiD image_id side/view')
    left = pd.to_numeric(df.left_eye_DR_Level,errors='coerce')
    right = pd.to_numeric(df.right_eye_DR_Level,errors='coerce')
    eye_grade = left.where(side.eq('l'), right)
    if eye_grade.isna().any() or not eye_grade.isin([0,1,2,3,4]).all():
        raise ValueError('Selected eye grades must be available as 0–4')
    overall = pd.to_numeric(df['Overall quality'],errors='coerce')
    if overall.isna().any() or not overall.isin([0,1]).all():
        raise ValueError('Unrecognized Overall quality coding')
    out = pd.DataFrame({'split': split_name,
                        'patient_id': df.patient_id.astype(str).map(lambda s: f'deepdrid_{s}'),
                        'image_id': df.image_id.astype(str),
                        'eye': side.map({'l':'left','r':'right'}),
                        'grade': eye_grade.astype(int),
                        'quality_flag_original': overall.astype(int),
                        'clarity': pd.to_numeric(df.Clarity, errors='coerce'),
                        'field_definition': pd.to_numeric(df['Field definition'],errors='coerce'),
                        'artifact': pd.to_numeric(df.Artifact,errors='coerce')})
    if out.image_id.duplicated().any(): raise ValueError('Duplicate image_id in the split')
    return out


def audit(train, validation):
    a=normalize_labels(train,'official_train')
    b=normalize_labels(validation,'official_validation')
    common=set(a.patient_id)&set(b.patient_id)
    if common: raise ValueError(f'Official split patient overlap detected: {len(common)}')
    results={}
    for name,df in [('train',a),('validation',b)]:
        tab=pd.crosstab(df.quality_flag_original,df.grade)
        results[name]={
            'images':len(df), 'patients':df.patient_id.nunique(),
            'eyes':df[['patient_id','eye']].drop_duplicates().shape[0],
            'quality_vs_eye_grade':{str(quality):{str(grade):int(n) for grade,n in row.items()}
                                    for quality,row in tab.to_dict(orient='index').items()},
            'grade_counts':{str(int(k)):int(v) for k,v in df.grade.value_counts().sort_index().items()},
            'quality_flag_counts':{str(int(k)):int(v) for k,v in df.quality_flag_original.value_counts().sort_index().items()},
            'missing_component_quality':int(df[['clarity','field_definition','artifact']].isna().any(axis=1).sum())}
    return results


def main():
    p=argparse.ArgumentParser(description='Verify local official DeepDRiD training and validation metadata')
    p.add_argument('--train',type=Path,required=True)
    p.add_argument('--validation',type=Path,required=True)
    p.add_argument('--out',type=Path,default=Path('deepdrid_metadata_audit.json'))
    args=p.parse_args()
    report=audit(args.train,args.validation)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    print('Saved audit to',args.out)

if __name__=='__main__': main()
