import pandas as pd

def convert_to_matrix(label_to_score_list, exps):
    all_labels = set()
    for label_to_score in label_to_score_list:
        all_labels.update(label_to_score.keys())
    all_labels = sorted(all_labels)
    
    mat = [
        [
            label_to_score[label]
            for label in all_labels
        ]
        for label_to_score in label_to_score_list
    ]
    df = pd.DataFrame(
        data=mat,
        index=exps,
        columns=all_labels
    )
    return df

if __name__ == '__main__':
    main()
