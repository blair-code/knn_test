mod data_loader;
use data_loader::data_loader::{CSVLoader, DataLoader};
fn main() {
    let path = "./iris.data.transf";
    let csv_ldr = CSVLoader{
        delimiter: b',',
        predictor_column_index: 4,
        has_headers: false
    };

    let (features, predictors) = csv_ldr.load_all_samples_from_disk(vec![path]);
    println!("{:?}", features);
    
}
