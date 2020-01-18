### Assuming this gets you real and fake data

# Real data
x.data.resize_as_(images).copy_(images)
y_pred = D(x)
y.data.resize_(current_batch_size).fill_(1)

# Fake data
z.data.resize_(current_batch_size, param.z_size, 1, 1).normal_(0, 1)
fake = G(z)
x_fake.data.resize_(fake.data.size()).copy_(fake.data)
y_pred_fake = D(x_fake.detach()) # For generator step do not detach
y2.data.resize_(current_batch_size).fill_(0)


### Standard GAN (non-saturating)

# Use torch.nn.Sigmoid() as last layer in discriminator

criterion = torch.nn.BCELoss()

# Real data Discriminator loss
errD_real = criterion(y_pred, y)
errD_real.backward()

# Fake data Discriminator loss
errD_fake = criterion(y_pred_fake, y2)
errD_fake.backward()

# Generator loss
errG = criterion(y_pred_fake, y)
errG.backward()


### Relativistic Standard GAN

# No sigmoid activation in last layer of discriminator because BCEWithLogitsLoss() already adds it

BCE_stable = torch.nn.BCEWithLogitsLoss()

# Discriminator loss
errD = BCE_stable(y_pred - y_pred_fake, y)
errD.backward()

# Generator loss (You may want to resample again from real and fake data)
errG = BCE_stable(y_pred_fake - y_pred, y)
errG.backward()


### Relativistic average Standard GAN

# No sigmoid activation in last layer of discriminator because BCEWithLogitsLoss() already adds it

BCE_stable = torch.nn.BCEWithLogitsLoss()

# Discriminator loss
errD = ((BCE_stable(y_pred - torch.mean(y_pred_fake), y) + BCE_stable(y_pred_fake - torch.mean(y_pred), y2))/2
errD.backward()

# Generator loss (You may want to resample again from real and fake data)
errG = ((BCE_stable(y_pred - torch.mean(y_pred_fake), y2) + BCE_stable(y_pred_fake - torch.mean(y_pred), y))/2
errG.backward()


### Relativistic average LSGAN

# No activation in discriminator

# Discriminator loss
errD = (torch.mean((y_pred - torch.mean(y_pred_fake) - y) ** 2) + torch.mean((y_pred_fake - torch.mean(y_pred) + y) ** 2))/2
errD.backward()

# Generator loss (You may want to resample again from real and fake data)
errG = (torch.mean((y_pred - torch.mean(y_pred_fake) + y) ** 2) + torch.mean((y_pred_fake - torch.mean(y_pred) - y) ** 2))/2
errG.backward()


### Relativistic average HingeGAN

# No activation in discriminator

# Discriminator loss
errD = (torch.mean(torch.nn.ReLU()(1.0 - (y_pred - torch.mean(y_pred_fake)))) + torch.mean(torch.nn.ReLU()(1.0 + (y_pred_fake - torch.mean(y_pred)))))/2
errD.backward()

# Generator loss  (You may want to resample again from real and fake data)
errG = (torch.mean(torch.nn.ReLU()(1.0 + (y_pred - torch.mean(y_pred_fake)))) + torch.mean(torch.nn.ReLU()(1.0 - (y_pred_fake - torch.mean(y_pred)))))/2
errG.backward()
