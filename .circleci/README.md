#README

This is the code that defines our CI/CD flow on CircleCI

#### dev
Deploying to `dev` is automatic for all branches.
However only the `master` branch is kept running at all times. All other branches are turned off by default (see the `stop-instance` step in config.yml). To allow traffic to pass, one must "start" the instance in the appEngine console.
We have chosen to do this to keep AppEngine costs down.

Each **branch** that gets deployed and has been started can be reached as
https://<branchname>-dot-open-targets-library.appspot.com

The version that gets deployed in production is instead versioned by github **tag**.
production add a `prod-*` tag.

#### production
Any `prod-*` tag gets deployed to https://link.opentargets.io). However:
- Deploying to `production` only happens with manual approval in CircleCI
- Traffic is not migrated automatically and needs to be done manually in each regional project.

Each **tag** that gets deployed can be reached as
https://<tagname>-dot-open-targets-library.appspot.com


### You might want to:

* change ES_MAIN_URL and ES_GRAPH_URLvariables in the `config.yml` to reflect the ES for each project.


### To trigger a production deployment:
```sh
TAG=$( echo prod-`date "+%Y%m%d-%H%M"`); git tag $TAG && git push origin $TAG
```

## What about VERSION?
Deployment does not depend on a change in the VERSION file, but a different version will trigger a new google endpoint deployment
