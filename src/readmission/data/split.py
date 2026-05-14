from sklearn.model_selection import train_test_split


def split_features_target(data, target_column: str):
    return data.drop(columns=[target_column]), data[target_column]


def train_test(data, target_column: str, test_size: float = 0.2, random_state: int = 42):
    return train_test_split(
        data.drop(columns=[target_column]),
        data[target_column],
        test_size=test_size,
        random_state=random_state,
        stratify=data[target_column],
    )

