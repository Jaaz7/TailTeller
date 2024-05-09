# just a file to automate my dockerfile image to the container at
# tailteller app in heroku

#!/bin/bash
APP_NAME="tailteller"
heroku container:login
heroku container:push web -a $APP_NAME
heroku container:release web -a $APP_NAME
echo "Deployment completed"