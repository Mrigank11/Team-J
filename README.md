# Team-J
## Usage

### Class prediction
To predict class, use `src/class_prediction.py`. For example:
```bash
./class_prediction.py --movie-id 356 --persistence-folder tmp
```
The above command will make predictions for the movie with id=356 i.e. Forrest Gump.

For more information on arguments, run `./src/class_prediction.py -h`

### Request prediction
To predict request count using xgboost, first make sure that `xgboost` is installed:
```bash
pip install xgboost
```
Then use `./request_prediction.py`.
```
./request_prediction.py -pf tmp -g Action, Adventure
```
