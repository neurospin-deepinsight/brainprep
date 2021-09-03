%-----------------------------------------------------------------------
% Job saved on 26-Mar-2021 15:27:40 by cfg_util (rev $Rev: 7345 $)
% spm SPM - Unknown
% cfg_basicio BasicIO - Unknown
%-----------------------------------------------------------------------
%% https://www.linuxquestions.org/questions/slackware-14/libreoffice-libfontconfig-so-1-undefined-symbol-ft_done_mm_var-4175665794/

disp('brainprep CAT12 VBM longitudinal')
if isempty(which('spm')),
     throw(MException('SPMCheck:NotFound', 'SPM not in matlab path'));
end
[name, version] = spm('ver');
fprintf('SPM version: %s Release: %s\n',name, version);
fprintf('SPM path: %s\n', which('spm'));

matlabbatch{{1}}.spm.tools.cat.long.datalong.subjects = {{{{
    {anat_file}
}}}}';
matlabbatch{{1}}.spm.tools.cat.long.longmodel = 1;
matlabbatch{{1}}.spm.tools.cat.long.nproc = 2;
matlabbatch{{1}}.spm.tools.cat.long.opts.tpm = {{'{tpm_file}'}};
matlabbatch{{1}}.spm.tools.cat.long.opts.affreg = 'mni';
matlabbatch{{1}}.spm.tools.cat.long.opts.biasstr = 0.5;
matlabbatch{{1}}.spm.tools.cat.long.opts.biasacc = 0.5;

matlabbatch{{1}}.spm.tools.cat.long.extopts.segmentation.APP = 1070;
matlabbatch{{1}}.spm.tools.cat.long.extopts.segmentation.spm_kamap = 0;
matlabbatch{{1}}.spm.tools.cat.long.extopts.segmentation.LASstr = 0.5;
matlabbatch{{1}}.spm.tools.cat.long.extopts.segmentation.gcutstr = 2;
matlabbatch{{1}}.spm.tools.cat.long.extopts.segmentation.WMHC = 1;
matlabbatch{{1}}.spm.tools.cat.long.extopts.segmentation.NCstr = -Inf;
matlabbatch{{1}}.spm.tools.cat.long.extopts.segmentation.cleanupstr = 0.5;
matlabbatch{{1}}.spm.tools.cat.long.extopts.segmentation.BVCstr = 0.5;
matlabbatch{{1}}.spm.tools.cat.long.extopts.segmentation.SLC = 0;
matlabbatch{{1}}.spm.tools.cat.long.extopts.segmentation.mrf = 1;
matlabbatch{{1}}.spm.tools.cat.long.extopts.segmentation.restypes.optimal = [1 0.1];

matlabbatch{{1}}.spm.tools.cat.long.extopts.registration.dartel.darteltpm = {{'{darteltpm_file}'}};
matlabbatch{{1}}.spm.tools.cat.long.extopts.registration.shooting.regstr = 0.5;
matlabbatch{{1}}.spm.tools.cat.long.extopts.vox = 1.5;

matlabbatch{{1}}.spm.tools.cat.long.extopts.surface.pbtres = 0.5;
matlabbatch{{1}}.spm.tools.cat.long.extopts.surface.pbtmethod = 'pbt2x';
matlabbatch{{1}}.spm.tools.cat.long.extopts.surface.pbtlas = 0;
matlabbatch{{1}}.spm.tools.cat.long.extopts.surface.collcorr = 0;
matlabbatch{{1}}.spm.tools.cat.long.extopts.surface.reduce_mesh = 1;
matlabbatch{{1}}.spm.tools.cat.long.extopts.surface.vdist = 1.33333333333333;
matlabbatch{{1}}.spm.tools.cat.long.extopts.surface.scale_cortex = 0.7;
matlabbatch{{1}}.spm.tools.cat.long.extopts.surface.add_parahipp = 0.1;
matlabbatch{{1}}.spm.tools.cat.long.extopts.surface.close_parahipp = 0;

matlabbatch{{1}}.spm.tools.cat.long.extopts.admin.experimental = 0;
matlabbatch{{1}}.spm.tools.cat.long.extopts.admin.new_release = 0;
matlabbatch{{1}}.spm.tools.cat.long.extopts.admin.lazy = 0;
matlabbatch{{1}}.spm.tools.cat.long.extopts.admin.ignoreErrors = 1;
matlabbatch{{1}}.spm.tools.cat.long.extopts.admin.verb = 2;
matlabbatch{{1}}.spm.tools.cat.long.extopts.admin.print = 2;

matlabbatch{{1}}.spm.tools.cat.long.output.surface = 0;
matlabbatch{{1}}.spm.tools.cat.long.output.surf_measures = 1;

matlabbatch{{1}}.spm.tools.cat.long.ROImenu.atlases.neuromorphometrics = 1;
matlabbatch{{1}}.spm.tools.cat.long.ROImenu.atlases.lpba40 = 0;
matlabbatch{{1}}.spm.tools.cat.long.ROImenu.atlases.cobra = 0;
matlabbatch{{1}}.spm.tools.cat.long.ROImenu.atlases.hammers = 0;
matlabbatch{{1}}.spm.tools.cat.long.ROImenu.atlases.ibsr = 0;
matlabbatch{{1}}.spm.tools.cat.long.ROImenu.atlases.aal3 = 0;
matlabbatch{{1}}.spm.tools.cat.long.ROImenu.atlases.mori = 0;
matlabbatch{{1}}.spm.tools.cat.long.ROImenu.atlases.anatomy = 0;
matlabbatch{{1}}.spm.tools.cat.long.ROImenu.atlases.julichbrain = 0;
matlabbatch{{1}}.spm.tools.cat.long.ROImenu.atlases.Schaefer2018_100Parcels_17Networks_order = 0;
matlabbatch{{1}}.spm.tools.cat.long.ROImenu.atlases.Schaefer2018_200Parcels_17Networks_order = 0;
matlabbatch{{1}}.spm.tools.cat.long.ROImenu.atlases.Schaefer2018_400Parcels_17Networks_order = 0;
matlabbatch{{1}}.spm.tools.cat.long.ROImenu.atlases.Schaefer2018_600Parcels_17Networks_order = 0;
matlabbatch{{1}}.spm.tools.cat.long.ROImenu.atlases.ownatlas = {{''}};

matlabbatch{{1}}.spm.tools.cat.long.output.GM.native = 1;
matlabbatch{{1}}.spm.tools.cat.long.output.GM.warped = 0;
matlabbatch{{1}}.spm.tools.cat.long.output.GM.mod = 1;
matlabbatch{{1}}.spm.tools.cat.long.output.GM.dartel = 0;
matlabbatch{{1}}.spm.tools.cat.long.output.WM.native = 1;
matlabbatch{{1}}.spm.tools.cat.long.output.WM.warped = 0;
matlabbatch{{1}}.spm.tools.cat.long.output.WM.mod = 1;
matlabbatch{{1}}.spm.tools.cat.long.output.WM.dartel = 0;
matlabbatch{{1}}.spm.tools.cat.long.output.CSF.native = 1;
matlabbatch{{1}}.spm.tools.cat.long.output.CSF.warped = 0;
matlabbatch{{1}}.spm.tools.cat.long.output.CSF.mod = 1;
matlabbatch{{1}}.spm.tools.cat.long.output.CSF.dartel = 0;
matlabbatch{{1}}.spm.tools.cat.long.output.ct.native = 0;
matlabbatch{{1}}.spm.tools.cat.long.output.ct.warped = 0;
matlabbatch{{1}}.spm.tools.cat.long.output.ct.dartel = 0;
matlabbatch{{1}}.spm.tools.cat.long.output.pp.native = 0;
matlabbatch{{1}}.spm.tools.cat.long.output.pp.warped = 0;
matlabbatch{{1}}.spm.tools.cat.long.output.pp.dartel = 0;
matlabbatch{{1}}.spm.tools.cat.long.output.WMH.native = 0;
matlabbatch{{1}}.spm.tools.cat.long.output.WMH.warped = 0;
matlabbatch{{1}}.spm.tools.cat.long.output.WMH.mod = 0;
matlabbatch{{1}}.spm.tools.cat.long.output.WMH.dartel = 0;
matlabbatch{{1}}.spm.tools.cat.long.output.SL.native = 0;
matlabbatch{{1}}.spm.tools.cat.long.output.SL.warped = 0;
matlabbatch{{1}}.spm.tools.cat.long.output.SL.mod = 0;
matlabbatch{{1}}.spm.tools.cat.long.output.SL.dartel = 0;
matlabbatch{{1}}.spm.tools.cat.long.output.TPMC.native = 0;
matlabbatch{{1}}.spm.tools.cat.long.output.TPMC.warped = 0;
matlabbatch{{1}}.spm.tools.cat.long.output.TPMC.mod = 0;
matlabbatch{{1}}.spm.tools.cat.long.output.TPMC.dartel = 0;
matlabbatch{{1}}.spm.tools.cat.long.output.atlas.native = 0;
matlabbatch{{1}}.spm.tools.cat.long.output.atlas.warped = 0;
matlabbatch{{1}}.spm.tools.cat.long.output.atlas.dartel = 0;
matlabbatch{{1}}.spm.tools.cat.long.output.label.native = 0;
matlabbatch{{1}}.spm.tools.cat.long.output.label.warped = 0;
matlabbatch{{1}}.spm.tools.cat.long.output.label.dartel = 0;
matlabbatch{{1}}.spm.tools.cat.long.output.labelnative = 1;
matlabbatch{{1}}.spm.tools.cat.long.output.bias.native = 0;
matlabbatch{{1}}.spm.tools.cat.long.output.bias.warped = 1;
matlabbatch{{1}}.spm.tools.cat.long.output.bias.dartel = 0;
matlabbatch{{1}}.spm.tools.cat.long.output.las.native = 0;
matlabbatch{{1}}.spm.tools.cat.long.output.las.warped = 0;
matlabbatch{{1}}.spm.tools.cat.long.output.las.dartel = 0;
matlabbatch{{1}}.spm.tools.cat.long.output.jacobianwarped = 0;
matlabbatch{{1}}.spm.tools.cat.long.output.warps = [1 1];
matlabbatch{{1}}.spm.tools.cat.long.output.rmat = 0;

matlabbatch{{1}}.spm.tools.cat.long.longTPM = 0;
matlabbatch{{1}}.spm.tools.cat.long.modulate = 1;
matlabbatch{{1}}.spm.tools.cat.long.dartel = 0;
matlabbatch{{1}}.spm.tools.cat.long.delete_temp = 0;