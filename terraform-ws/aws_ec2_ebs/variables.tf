variable "region" {
    default = "ap-south-1"
}
variable "instance_type" {
    default = "t2.micro"
}
variable "ebs_size" {
    type = number
    default = 1
}

