import pickle
import numpy as np
import matplotlib.pyplot as plt
import math
from math import sqrt

"""
-> userX.pickle (list of dictionaries)    
    -> dictionary => keys: "session_names" ve values "session_probabilities"
    -> dictionary => keys: "session_names" ve values "labels"Labellarda 0: legal, 1:illegal.
Önce tek kullanıcıya odaklan, yazdığın fonksiyonlar genelleştirilebilir olsun.
"""

with open("user29.pickle",'rb') as handle:
    user_data = pickle.load(handle)

probability_dict = user_data[0]
labels_dict = user_data[1]
#Session names'leri başka bir yerde bastırıp buraya her biri için array içine yazdım.
session_names7=['session_3748528962', 'session_1081274523', 'session_8999428064', 'session_2773924704', 'session_0147719489', 'session_2318637737', 'session_1881356230', 'session_5037856261', 'session_5026661112', 'session_9362179236', 'session_5884843843', 'session_9067768063', 'session_1244242475', 'session_9495997885', 'session_6581338506', 'session_9278196183', 'session_4163238472', 'session_2211907871', 'session_1063325046', 'session_6471969473', 'session_7157064073', 'session_3376026513', 'session_6556036411', 'session_6106477620', 'session_4339907415', 'session_0812869833', 'session_4423579184', 'session_8810377918', 'session_0966487358', 'session_5528609206', 'session_7370840078', 'session_1503605581', 'session_8883281313', 'session_6013544622', 'session_3582091129', 'session_6738388054', 'session_8140667141', 'session_9877506321', 'session_4844871120', 'session_7873949379', 'session_7780444958', 'session_7358104277', 'session_6129550717', 'session_6490827344', 'session_0984393142', 'session_9689083079', 'session_3319050185', 'session_5497122814', 'session_8769574094', 'session_8927734902', 'session_1474073005', 'session_2671379191', 'session_2590894741', 'session_6419217298', 'session_1708671050', 'session_9880892041', 'session_5123812030', 'session_6638271928', 'session_3876004338', 'session_9100599511', 'session_3973707215', 'session_5289449664', 'session_3269754223', 'session_2715537964', 'session_0244684556', 'session_5483480261', 'session_7451527108', 'session_8989374051', 'session_1806185715', 'session_4037858026', 'session_2691409086', 'session_7933738052', 'session_6617825429']
session_names9=['session_5745729038', 'session_0867569021', 'session_1814762834', 'session_1067162598', 'session_1177848198', 'session_9495657954', 'session_7729762375', 'session_0626697371', 'session_8602611959', 'session_7861130331', 'session_9821034386', 'session_7935100368', 'session_2092170870', 'session_6228883788', 'session_4322210317', 'session_3452570104', 'session_8719700167', 'session_4066780045', 'session_4948057062', 'session_5496662888', 'session_9000198916', 'session_0233596484', 'session_3735537656', 'session_1334666365', 'session_3524810503', 'session_1498771725', 'session_7896781089', 'session_9316476581', 'session_1428731346', 'session_2592273518', 'session_8399176672', 'session_9916663391', 'session_0510101673', 'session_4088341904', 'session_4949372670', 'session_3926840201', 'session_9418948998', 'session_9904426178', 'session_4243186683', 'session_3919629858', 'session_1970148824', 'session_7422282952', 'session_4619254586', 'session_3561215335', 'session_0249395771', 'session_8742024614', 'session_7555995161', 'session_4962669721', 'session_6845822655', 'session_1388817097', 'session_2541149709', 'session_6399026328', 'session_7422270211', 'session_3828692242', 'session_2386499713', 'session_7395469781', 'session_5499827097', 'session_6448386600', 'session_1281792837', 'session_7103728864', 'session_2760097341', 'session_6951064037', 'session_0815666379', 'session_3277767566', 'session_9057489846', 'session_4269912456']
session_names12=['session_6237876052', 'session_0195566274', 'session_9899715696', 'session_0611188910', 'session_6446737316', 'session_4876159319', 'session_5256432882', 'session_0166199610', 'session_8291380356', 'session_9924186646', 'session_8199358611', 'session_0172989910', 'session_3226676976', 'session_4905082660', 'session_2726751248', 'session_8271683052', 'session_7168735362', 'session_7907939720', 'session_9506925023', 'session_2177337196', 'session_0126772600', 'session_6167829880', 'session_7575485743', 'session_3360873587', 'session_1928096865', 'session_2116060291', 'session_3315925736', 'session_7976664538', 'session_3267760096', 'session_9973193301', 'session_9816302916', 'session_3230049516', 'session_3683562482', 'session_8439712811', 'session_4970622399', 'session_6695269196', 'session_6142373482', 'session_4066543084', 'session_0831009063', 'session_0718520474', 'session_9693789671', 'session_1196335015', 'session_3958419196', 'session_3678562948', 'session_1843905203', 'session_7562248450', 'session_8689827198', 'session_4091639526', 'session_7454853209', 'session_5052301661', 'session_1656689994', 'session_4383618618', 'session_6468388206', 'session_5035500864', 'session_0170625567', 'session_6229277499', 'session_7684485021', 'session_9072596713', 'session_3657391768', 'session_7772954143', 'session_5518980455', 'session_9932324920', 'session_3487526409', 'session_9387370115', 'session_5577188457', 'session_5269540222', 'session_7614908860', 'session_7580547206', 'session_7914718036', 'session_8950902312', 'session_3807007352', 'session_9203064974', 'session_2923450227', 'session_6835186342', 'session_6965771386', 'session_5771189096', 'session_8701118698', 'session_4498932663', 'session_5352520893', 'session_8674602823', 'session_0893120027', 'session_6304072815', 'session_8014286229', 'session_8361853378', 'session_3387344935', 'session_5088690608', 'session_8680283238', 'session_6721137284', 'session_7583047056', 'session_8200552592', 'session_6186753676', 'session_5747551041', 'session_5346999982', 'session_9839818954', 'session_4414280148', 'session_7489987884', 'session_7251443361', 'session_5476317070', 'session_0172860263', 'session_1178629549', 'session_6974460768', 'session_3571551367', 'session_5466872812', 'session_5046103917', 'session_4157921188']
session_names15=['session_2613999059', 'session_1824070206', 'session_3791802350', 'session_8557723888', 'session_4896261465', 'session_1149922141', 'session_8091715457', 'session_6896514118', 'session_0003960194', 'session_9505884451', 'session_3102058551', 'session_7989414599', 'session_9795347408', 'session_4290136840', 'session_0612796637', 'session_3343831517', 'session_6869635181', 'session_8671492463', 'session_7323696974', 'session_6464183153', 'session_3879503861', 'session_9501338931', 'session_6679825465', 'session_3814755566', 'session_1649882646', 'session_1567411744', 'session_6189139492', 'session_4480398495', 'session_8423329764', 'session_2203453961', 'session_9921687223', 'session_7944252331', 'session_4762909710', 'session_1618522149', 'session_3051589624', 'session_0864574884', 'session_6237417362', 'session_5023081923', 'session_6303895598', 'session_7818234002', 'session_9767538073', 'session_7690305433', 'session_5700842190', 'session_8769983957', 'session_2953697423', 'session_0128859274', 'session_6657360579', 'session_1876247964', 'session_8934034457', 'session_7979911880', 'session_2730276507', 'session_0305187451', 'session_7800269296', 'session_7394343234', 'session_7071815036', 'session_7035039573', 'session_0719528275', 'session_3014121416', 'session_8351091371', 'session_1316321566', 'session_3155649031', 'session_7779665504', 'session_6751955279', 'session_3704248627', 'session_3641521673', 'session_6349231133', 'session_6873559742', 'session_0326724732', 'session_7331556356', 'session_8691825471', 'session_6705777193', 'session_6112730640', 'session_7967310508', 'session_1324126678', 'session_1854610785', 'session_5695398064', 'session_4386624259', 'session_5594141097', 'session_4687655362', 'session_6320970139', 'session_9729063367', 'session_9983042278', 'session_5958053029', 'session_6102555994', 'session_3393326634', 'session_7096197451', 'session_2694965772', 'session_9809839685', 'session_5806761857', 'session_4098569257', 'session_8740037149', 'session_2931074825', 'session_5071023384', 'session_6568302079', 'session_0647531474', 'session_1016856509', 'session_0510406466', 'session_1750509621', 'session_1968614536', 'session_2099251918', 'session_4119002624', 'session_7471072415', 'session_5269315187', 'session_7800181258', 'session_6509364535', 'session_3077149461', 'session_6871552747', 'session_0861337889', 'session_9411417025', 'session_0280517098', 'session_0797370695', 'session_4659905162', 'session_3603344105', 'session_2375808482', 'session_5312030236']
session_names16=['session_2926423726', 'session_9830083627', 'session_6088382546', 'session_1271171766', 'session_5556934089', 'session_3367539943', 'session_9877366244', 'session_8884379611', 'session_8439483653', 'session_5041518490', 'session_5197680121', 'session_8811009703', 'session_4238071281', 'session_4026966824', 'session_1535403881', 'session_4265171670', 'session_0408822104', 'session_8862058156', 'session_3974162531', 'session_2853115772', 'session_9791921163', 'session_0482518942', 'session_8069308945', 'session_1976172513', 'session_5685066201', 'session_5319036655', 'session_2636442787', 'session_4432555629', 'session_3735635217', 'session_8389698709', 'session_0064281061', 'session_2977418379', 'session_1667382627', 'session_5889567297', 'session_5127880135', 'session_0770188364', 'session_5030324559', 'session_1256331958', 'session_7744249581', 'session_8115384066', 'session_3071944492', 'session_0953745782', 'session_8364143294', 'session_3573257812', 'session_5576759361', 'session_1427407488', 'session_6179037141', 'session_0005840196', 'session_2683617861', 'session_5572048254', 'session_3206758270', 'session_2494483407', 'session_7992883230', 'session_1653923890', 'session_3726985009', 'session_4224530762', 'session_8583637812', 'session_9705613967', 'session_4098547958', 'session_2083267904', 'session_7850373093', 'session_6119506741', 'session_6915176151', 'session_7618706636', 'session_5823371082', 'session_2194913037', 'session_8857212561', 'session_9477322863', 'session_9794510496', 'session_8823455059', 'session_4360886220', 'session_8819855375', 'session_0844521549', 'session_1717253391', 'session_3067782503', 'session_3349837388', 'session_4615055511', 'session_0668352196', 'session_0859353781', 'session_8776482178', 'session_0025450757', 'session_1386411131', 'session_3583196291', 'session_4506008213', 'session_0083463746', 'session_8070684894', 'session_3780421656', 'session_1484409764', 'session_7986274484', 'session_3656309644', 'session_0148970615', 'session_0223955219', 'session_6100708780', 'session_5771275112', 'session_6775680298', 'session_0155746039', 'session_5968190840', 'session_3348997730', 'session_8770654755', 'session_9919383972', 'session_1619535766', 'session_8653789319', 'session_8636893699', 'session_6835145359', 'session_4600074976', 'session_4794437261']
session_names20=['session_9708106289', 'session_2336058051', 'session_2532367006', 'session_3379861047', 'session_5982655120', 'session_5453779030', 'session_1924699326', 'session_4254477956', 'session_8833850952', 'session_8658186994', 'session_3236365486', 'session_5205618080', 'session_9753314758', 'session_5340758381', 'session_0799578885', 'session_1868010893', 'session_5938861210', 'session_3005678154', 'session_2368259027', 'session_4339216244', 'session_8687623726', 'session_9064183032', 'session_5321706137', 'session_9395770534', 'session_9160177818', 'session_3659572440', 'session_4804227584', 'session_0379715237', 'session_9086606611', 'session_9646127676', 'session_6706849000', 'session_5445638904', 'session_0512046694', 'session_7589485664', 'session_3684692596', 'session_5291244662', 'session_3482932637', 'session_8348355115', 'session_9740386657', 'session_2320125003', 'session_1468258531', 'session_6710819502', 'session_0101735014', 'session_7259628766', 'session_1311858241', 'session_2861116304', 'session_8627857957', 'session_6555326593', 'session_0593223632', 'session_1273150073']
session_names21=['session_4741380705', 'session_8452199606', 'session_0733542929', 'session_2383353199', 'session_6723163956', 'session_6350129821', 'session_1376706431', 'session_0334337435', 'session_2778964895', 'session_6803917267', 'session_0900979755', 'session_3290502212', 'session_0489826159', 'session_2305469177', 'session_7288721890', 'session_3567705649', 'session_3473124800', 'session_9913386649', 'session_3318991223', 'session_0200062241', 'session_3985625607', 'session_5226344095', 'session_8957360206', 'session_9938110038', 'session_6747468371', 'session_4803913319', 'session_0080153528', 'session_7800887945', 'session_1065465945', 'session_3382231612', 'session_8675020847', 'session_2472958094', 'session_3735387695', 'session_2776997709', 'session_5158497089', 'session_7208965815', 'session_8067504883', 'session_0486100880', 'session_7053852427', 'session_7756203951', 'session_2681498481', 'session_6410652005', 'session_1193964670', 'session_7953065762', 'session_0477165267', 'session_5035531254', 'session_8534613968', 'session_5764163634', 'session_2037079652', 'session_2547686354', 'session_4873496968', 'session_1809941518', 'session_6891188294', 'session_4282931799', 'session_9814859818', 'session_6905592155', 'session_2476629136', 'session_8456906043', 'session_5896454946']
session_names23=['session_4088744338', 'session_4354218638', 'session_3760606976', 'session_1414301302', 'session_5938044199', 'session_9579651012', 'session_0750796173', 'session_3436410606', 'session_0697506648', 'session_4484768740', 'session_9648839315', 'session_5130764696', 'session_0139259699', 'session_7131856482', 'session_3193098342', 'session_7778589669', 'session_5760480696', 'session_9034407980', 'session_5567012419', 'session_8916797638', 'session_5796305508', 'session_7756346661', 'session_9956793065', 'session_1783392506', 'session_4144841412', 'session_8427974282', 'session_5138848212', 'session_7398972341', 'session_4736309542', 'session_5967488943', 'session_6575637744', 'session_3496233301', 'session_4998803513', 'session_3922157039', 'session_1799284692', 'session_6120487222', 'session_0590731055', 'session_5653345386', 'session_3173348385', 'session_8055688583', 'session_2806235778', 'session_0071280153', 'session_0542118784', 'session_8297767726', 'session_7144916399', 'session_8431086878', 'session_2676613576', 'session_2887480941', 'session_2020107805', 'session_3777853846', 'session_9896236015', 'session_6529033101', 'session_1581692104', 'session_9787004965', 'session_9759261305', 'session_2222178768', 'session_7760528986', 'session_5536570637', 'session_6479783256', 'session_7434938340', 'session_9307936288', 'session_6200753435', 'session_4185228735', 'session_8831138885', 'session_4199921692', 'session_9127984644', 'session_5652085012', 'session_8301590977', 'session_9038755287', 'session_5090921415', 'session_5222969532']
session_names29=['session_3147629890', 'session_5924200824', 'session_0508486473', 'session_5894660899', 'session_2617473265', 'session_1896830251', 'session_5746562839', 'session_1421795197', 'session_1967235442', 'session_8423566466', 'session_6982446047', 'session_7011327614', 'session_5623988336', 'session_3932398331', 'session_5290417223', 'session_9497021147', 'session_2290361531', 'session_0843115603', 'session_3161171774', 'session_4746240778', 'session_1164286300', 'session_9951071945', 'session_9046061162', 'session_1819563622', 'session_7418988874', 'session_8604010887', 'session_3172392634', 'session_9518245436', 'session_3163625589', 'session_0228122983', 'session_6007924250', 'session_8627031256', 'session_2189267771', 'session_7473209927', 'session_6989962465', 'session_4814782358', 'session_2064160756', 'session_2035629884', 'session_5415060346', 'session_8407883787', 'session_2540441733', 'session_9551087650', 'session_1076519237', 'session_0537801506', 'session_3839970917', 'session_8119180048', 'session_3660445831', 'session_1778570235', 'session_5487557068', 'session_9673398856', 'session_2046923002', 'session_3016761703', 'session_7648037009', 'session_0305606782', 'session_4262101552', 'session_0270940804', 'session_6056134961', 'session_0838891042', 'session_6527951269', 'session_9197162417', 'session_8082239300', 'session_0743065918', 'session_2007540735']
session_names35=['session_3635374508', 'session_7388997727', 'session_4518223786', 'session_2389761713', 'session_6480157843', 'session_1893250984', 'session_8639984402', 'session_7997263430', 'session_2272584671', 'session_5445301341', 'session_5527782146', 'session_0989732540', 'session_7370016891', 'session_8015250878', 'session_3097489009', 'session_2751066909', 'session_7768986100', 'session_4767254104', 'session_6295392963', 'session_2069574400', 'session_2689331954', 'session_5160736279', 'session_3101765401', 'session_7354123043', 'session_1306722148', 'session_0029922803', 'session_6479266110', 'session_8478632285', 'session_5016669083', 'session_4559429580', 'session_6425950736', 'session_9127961354', 'session_7667901628', 'session_3212035675', 'session_2751916214', 'session_6970752744', 'session_0493270999', 'session_1129865931', 'session_2878525110', 'session_9525615730', 'session_4481103124', 'session_8281638783', 'session_2426108754', 'session_7217479988', 'session_0111356050', 'session_7093950723', 'session_8185572921', 'session_8214714439', 'session_6816047521', 'session_3170574491', 'session_7326699927', 'session_9674716665', 'session_3028558496', 'session_2412921544', 'session_3389870646', 'session_3762712464', 'session_3981019566', 'session_9653209270', 'session_0462903577', 'session_6933816157', 'session_7100038229', 'session_6637610294', 'session_3696132790', 'session_3669155019', 'session_7437771065', 'session_3535166347', 'session_3876999904', 'session_0458723853', 'session_5517823185', 'session_1553613542', 'session_9716805694', 'session_2809105327', 'session_3203109772', 'session_8042510915', 'session_2585594441', 'session_5013714842', 'session_6051914984', 'session_7534749559', 'session_4289581476', 'session_9020534315', 'session_3182721337', 'session_7788563873', 'session_5248409822', 'session_7273363943', 'session_0841557171', 'session_7177007633', 'session_9629836105', 'session_9183184177', 'session_6497918859', 'session_6452949084', 'session_0376544801', 'session_6278669958', 'session_8728901791', 'session_2110642502', 'session_0402994139', 'session_5425983208', 'session_5690417333', 'session_5811752669', 'session_2363410078', 'session_2381827805', 'session_7970857766', 'session_4251225905', 'session_9667273084', 'session_4331334148', 'session_5931589642', 'session_8340206664', 'session_6829066411', 'session_3116416990']

train=[]    #Boş train matris oluşturdum sonra da eklemek için.
counter=1   #Her sessiondan train için legal kısmın %70'ini alabilmek için oluşturduğum counter
prob=[]
prob_counter=0
session_length=len(session_names29)
k=0   #Userın sessionlarını saymak için kullandığım sayaç
def feature_extraction(session_names):
while(k<len(session_names29)):
    xx=session_names29[k] #Sessionın ismini alıyorum
    k=k+1
    first_prob=probability_dict[xx] #Kayıtlı olasılıkları burada yeni bir arraye oluşturarak kullanıyorum
    first_prob = first_prob[~np.isnan(first_prob)]
    #Burada session için gelen olasılıkları 5li gruplayarak yeni bir olasılık sıralaması oluşturdum. Slice Window methodunu kullanmak için
    while(prob_counter<len(first_prob)-5):
        prob.append((first_prob[prob_counter]+first_prob[prob_counter+1]+first_prob[prob_counter+2]+first_prob[prob_counter+3]+first_prob[prob_counter+4])/5)
        prob_counter+=1
#   Burası user için olan sessionların olasılıklarını train arrayine atıyorum daha sonra sırayla kullanabilmek amacıyla.   
    if (labels_dict[xx]==0):
        counter=counter+1 #legal sessionların sayısını burada tutuyorum
        train.append(prob)
featc=0 #train data sayacı kullanımı kontrol etmek için
summation=0 # average bulmak için olasılıkların toplamını burada tutuyorum
length_train=0 #train için kullanılan tüm olasılıkların uzunluğu bu değişkende tutuldu
train_limit=7*counter/10 #Legal sessionların %70'lik kısmını kullanmak için bu limiti ayarladım.
while(featc<train_limit):    
    summation = sum(train[featc]) + summation
    length_train = len(train[featc]) + length_train
    featc=featc+1
average=summation/length_train
varcount=0 # average bulunmuştu bu değişken variance hesabında trainde kullanıcalak sessionları ayarlamak için kullanıldı.
temp=[]
varsum=0
while(varcount<train_limit):
    temp=train[varcount]
    varcount = varcount +1
    insidecount=0
    while(insidecount<len(temp)):
        input = temp[insidecount]
        varsum = (input - average) ** 2 + varsum
        insidecount = insidecount  + 1
        
S=math.sqrt(varsum / (length_train - 1)) # variance was calculated here
lam=0.3  #lambda default between 0<lambda<=0.3. I choose 0.3
h = 5   # default 3 and a parameter for EMWA Chart
n = 120  # number of all data 
ucl = average + h * S
lcl = average - h * S            
T=average
yudidit = 0 # doğru bilinen sessionların sayısının tutulduğu değişken
session_counter = 0 # sessionın sırasının tutulduğu counter.
session_names=[]
while(session_counter<session_length):    
    ss=session_names29[session_counter]
    prob=[]
    first_prob=probability_dict[ss]
    first_prob = first_prob[~np.isnan(first_prob)]
    while(prob_counter<len(first_prob)-5):
        prob.append((first_prob[prob_counter]+first_prob[prob_counter+1]+first_prob[prob_counter+2]+first_prob[prob_counter+3]+first_prob[prob_counter+4])/5)
        prob_counter+=1      
    uclg=[]
    lclg=[]
    uclg=np.ones(len(prob)) * ucl
    lclg=np.ones(len(prob)) * lcl * 0
    CL=[]
    CU=[]
    m=5 #subgroup number 
    f=1 #default değerleri kullanıldı
    k=1 #default değerleri kullanıldı
    CL.append(f * S / sqrt(m))#lower variables were showed there.This was first variable
    CU.append(f * S / sqrt(m))#upper variables were showed there.This was first variable
    i = 1
    generaltemp=0 #session için legal mi illegal mi diye tutan değişken
    temp0=0 #legal olan olasılık
    temp1=0 #illegal olan olasılık
    session_counter=session_counter+1
    for data in prob:
        xl = min(T , (CL[i-1] + data - (T - (k * S / sqrt(m)))))
        CL.append(xl)
        CU.append(max( T, T + (CL[i-1] + data - (T + (k * S / sqrt(m))))))
        if (CU[i] < ucl):
            temp0=temp0+1
        elif (CU[i] > ucl):
            temp1=temp1+1 
        i += 1  
    if temp0 > temp1:
        generaltemp=0
        
    else:
        generaltemp=1
    if generaltemp==labels_dict[ss]:
        yudidit=yudidit+1 #verilen data ile sonucum örtüşüyor ise değişken bir artıyor       
        
# ne kadarını bildiğini göstermek için değişkenleri bastırıyorum        
print(yudidit)
print(session_length)  