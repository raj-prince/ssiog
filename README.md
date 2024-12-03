# Synthetic Scale IO Generator (ssiog)

## Setup
### Create virtual environment
1. To create `python3 -m venv .venv`
2. To activate `source .venv/bin/activate`
3. To deactivate `deactivate`

### Install the requirement file
1. To ensure the latest pip `pip install --upgrade pip`
2. To install the requirements.txt `pip install --root-user-action ignore -r requirements.txt && pip cache purge`

### Install terraform
Follow [this](https://g3doc.corp.google.com/company/teams/cloud-octo/octo-dev/tools/terraform.md?cl=head#install-terraform-on-cloudtop) to install and make sure to select latest terraform version. To get the latest checkout [this](https://developer.hashicorp.com/terraform/install#linux).

### Terraform command
1. To init the resource: `terraform init`
2. To validate the configuration: `terraform validate'`
3. To plan: `terraform plan -out=/tmp/module.out`
4. To apply:`terraform apply /tmp/module.out`

