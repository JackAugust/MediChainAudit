# MediChainAudit
A Blockchain-based Auditable Scheme for Encrypted Medical Data Supporting Fuzzy Deduplication and Secure Sharing

## Organization Structure
### Code
#### FMLE
Our FMLE algorithm prototype is implemented in Go.
##### main.go
`go run main.go`

Fuzzy encryption and deduplication for testing 0.1% duplicate rate data ranging from 1MB to 4GB (comparison algorithms: AES-CTR, MLE with SHA256)

##### main-V2.go
Change `package main-V2` to `package main` and then `go run main.go`

Generate data of 64MB and 128MB sizes with a duplication rate ranging from `0.1` to `0.5`, and verify the duplicate detection accuracy of FMLE (compared to MLE).

#### PublicDataset
Verify the deduplication effect of FMLE on public datasets
[Medical records of 100 Synthea patients](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/VBEKZO)
[Medical records of 30K Synthea synthetic patients](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/BWDKXS)

##### Electronic Medical Record (EMR) Generation For 100 Synthea patients (30K is similar)
`python 01-generate_patient_reports.py`

Convert the dataset into electronic medical records (EMRs) for patients

##### Verify the FMLE Effect
`python 02-medical_deduplication_comparison.py`

Deduplication of Electronic Medical Records (Comparison with MLE)

### sc-testdata
Test results of four smart contracts (Upload、Share、Access、Audit) designed based on Fabric V2.2 (using Caliper V0.4.0)