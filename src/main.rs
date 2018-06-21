mod data_loader;
mod knn_shapley;

use data_loader::data_loader::{CSVLoader, DataLoader};
use knn_shapley::knn_shapley::calculate_knn_shapleys;
fn main() {
    let path = "./iris.data.transf";
    let csv_ldr = CSVLoader{
        delimiter: b',',
        predictor_column_index: 4,
        has_headers: false
    };

    let (features, predictors) = csv_ldr.load_all_samples_from_disk(vec![path]);
    
    let shapleys = calculate_knn_shapleys(&features, &predictors, &features, &predictors, 3);
    println!("{:?}", shapleys);
    
}
