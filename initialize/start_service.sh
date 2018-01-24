
# Start service with `nohup sh start_service.sh &`
# OR add to cron `@reboot . /home/squirrel/squirrel-watch/initialize/start_service.sh`
LOG_DIR="${HOME}/squirrel-watch/initialize"

source ${HOME}/.bash_profile
source deactivate
source activate squirrel
cd ${HOME}/squirrel-watch/src

export KERAS_BACKEND=tensorflow
python app.py > $LOG_DIR/squirrel.log 2>&1
