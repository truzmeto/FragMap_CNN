# VMD FragMap Plugin
#
# Copyright (c) 2016-2018 SilcsBio, LLC. All Rights Reserved.
#
# Any unauthorized reproduction, use, or modification of this work
# is prohibited.
#
# Contact:
#
# SilcsBio, LLC
# 8 Market Place
# Baltimore, MD, 21202
# 410-929-5783
# info@silcsbio.com
# http://www.silcsbio.com
#
# This work contains code derived from the VMD Prefs plugin written by
# Christopher G. Mayne (http://www.ks.uiuc.edu/Research/vmd/plugins/vmdprefs/)
# at University of Illinois, Urbana-Champaign.
#
# This work contains code derived from the pdbbfactor.tcl written by
# Justin Gullingsrud (http://www.ks.uiuc.edu/Research/vmd/script_library/scripts/pdbbfactor/pdbbfactor.tcl)
#

# package setup
package provide silcs 1.0

namespace eval ::silcs:: {

    variable w

    variable selection "(all)"
    variable prefix ""              # map prefix
    variable mapdir "2b_gen_maps"   # map folder
    variable liganddir "mol2"       # ligand folder
    variable mcsilcsdir "mcsilcs"   # mc-silcs folder
    variable maplevel_lower -7
    variable maplevel_upper 4
    variable fragmaps_available {
        apolar {
            label "Generic Apolar Map"
            color 12 # green
            visible 1
            cutoff -1.0
        }
        hbdon {
            label "Generic Donor Map"
            color 0 # blue
            visible 1
            cutoff -1.0
        }
        hbacc {
            label "Generic Acceptor Map"
            color 1 # red
            visible 1
            cutoff -1.0
        }
        meoo {
            label "Methanol Oxygen Map"
	    color 14 # ochre
	    visible 1
            cutoff -1.0
        }
        mamn {
            label "Methylammonium Nitrogen Map"
            color 10 # cyan
            visible 1
            cutoff -1.0
        }
        acec {
            label "acec ? Map"
            color 3 # orange
            visible 1
            cutoff -1.0
        }
	apolarP {
            label "Generic Apolar Pred."
	    color 5 # tan
            visible 1
            cutoff -1.0
        }
        hbdonP {
            label "Generic Donor Pred."
	    color 11 # purple
            visible 1
            cutoff -1.0
        }
        hbaccP {
            label "Generic Acceptor Pred."
	    color 19 # green2
            visible 1
            cutoff -1.0
        }
        meooP {
            label "Methanol Oxygen Pred."
            color 12 # lime
	    visible 1
            cutoff -1.0
        }
        mamnP {
            label "Methylammonium Nitrogen Pred."
	    color 9 # pink
	    visible 1
            cutoff -1.0
        }
        acecP {
            label "acec ? Pred."
	    color 15 # iceblue
	    visible 1
            cutoff -1.0
        }
    }
    variable fragmaps_enabled {
        apolar
	hbdon
	hbacc
	meoo
	mamn
	acec
	apolarP
	hbdonP
	hbaccP
	meooP
	mamnP
	acecP
    }
    variable pocketrep -1
}

proc fragmap {} {
    # global cmd for launching the gui
    return [eval ::silcs::fragmap]
}

proc ::silcs::fragmap {} {

    puts stdout "   \nInfo) FragMap Tool v1.0 by SilcsBio.\
                    \nInfo) http://www.silcsbio.com/ \
                    \nInfo) Contact info@silcsbio.com for support.\n"

    variable w

    # initialization procedure
    ::silcs::init

    # setup the GUI window
    if { [winfo exists .fragmap] } {
        wm deiconify .fragmap
        return
    }

    set w [toplevel ".fragmap"]
    wm title $w "FragMap Tools"

    # allow .fragmap to expand/contract with the window
    grid columnconfigure $w 0 -weight 1
    grid rowconfigure $w 0 -weight 1

    # set a default initial geometry
    wm geometry $w 810x820

    # build a high level frame (hlf)
    # everything will be stored inside this frame
    ttk::frame $w.hlf
    grid $w.hlf -column 0 -row 0 -sticky nswe
    # allow the hlf to expand/contract appropriately
    grid columnconfigure $w.hlf 0 -weight 1
    grid rowconfigure $w.hlf {2} -weight 1

    #
    # header panel
    #
    ttk::frame $w.hlf.header -padding 0
    label $w.hlf.header.label -text "VMD FragMap Tools

SilcsBio, 2019 - http://silcsbio.com" -anchor center -justify center -background #3857a5 -foreground white -font bold
    pack $w.hlf.header.label -expand 1 -fill both -padx 4 -pady 4
    grid $w.hlf.header -column 0 -row 0 -columnspan 1 -sticky nswe
    grid columnconfigure $w.hlf.header.label 0 -weight 1

    # build / grid a separator
    ttk::separator $w.hlf.sep1 -orient horizontal
    grid $w.hlf.sep1 -column 0 -row 1 -sticky nswe -pady 2 -padx 2

    #
    # Main panel
    #
    ttk::notebook $w.hlf.nb
    grid $w.hlf.nb -column 0 -row 2 -sticky nswe

    # main menu tab
    ttk::frame $w.hlf.nb.main
    #$w.hlf.nb add $w.hlf.nb.main -text "Main"
    set main $w.hlf.nb.main

    ttk::frame $main.frame
    grid $main.frame -column 0 -row 0 -sticky nwse -padx 10 -pady 10
    grid columnconfigure $main 0 -weight 1
    set entries {
        {var "mapdir" label "Map folder:" tv ::silcs::mapdir}
        {var "prefix" label "Map prefix:" tv ::silcs::prefix}
        {var "ligand" label "Ligand folder:" tv ::silcs::liganddir}
        {var "mcsilcs" label "MC-SILCS folder:" tv ::silcs::mcsilcsdir}
    }

    set nelem [llength $entries]
    for {set i 0} {$i < $nelem} {incr i} {
        set entry [lindex $entries $i]
        set label [dict get $entry label]
        set var [dict get $entry var]
        set tv [dict get $entry tv]

        ttk::label $main.frame.${var}_label -text $label -anchor w
        ttk::entry $main.frame.$var -textvariable $tv

        grid $main.frame.${var}_label -column 0 -row $i -sticky nswe
        grid $main.frame.$var -column 1 -row $i -sticky nswe
    }
    grid columnconfigure $main.frame 1 -weight 1

    # fragmap locations tab
    ttk::frame $w.hlf.nb.locations
    $w.hlf.nb add $w.hlf.nb.locations -text "FragMap Locations"
    set loc $w.hlf.nb.locations

    ttk::frame $loc.frame
    grid $loc.frame -column 2 -row 2 -sticky ns -padx 10 -pady 10

    # add mapdir and prefix to fragmap locations tab
    set entries {
        {var "mapdir" label "Map folder:" tv ::silcs::mapdir}
        {var "prefix" label "Map prefix:" tv ::silcs::prefix}
    }

    set nelem [llength $entries]
    set offset $nelem
    for {set i 0} {$i < $nelem} {incr i} {
        set entry [lindex $entries $i]
        set label [dict get $entry label]
        set var [dict get $entry var]
        set tv [dict get $entry tv]

        ttk::label $loc.frame.${var}_label -text $label -anchor w
        ttk::entry $loc.frame.$var -textvariable $tv

        grid $loc.frame.${var}_label -column 0 -row $i -sticky nswe
        grid $loc.frame.$var -column 1 -row $i -sticky nswe
    }

    set fragmaps_enabled $::silcs::fragmaps_enabled
    set fragmaps_available $::silcs::fragmaps_available
    set nelem [llength $fragmaps_enabled]
    for {set i 0} {$i < $nelem} {incr i} {
        set key [lindex $fragmaps_enabled $i]
        set fragmap [dict get $fragmaps_available $key]
        set label [dict get $fragmap label]
        set tv ${key}_fragmap_filename
        set idx [expr $i + $offset]

        ttk::label $loc.frame.${key}_label -text $label -anchor w
        ttk::entry $loc.frame.$key -textvariable ::silcs::$tv -width 50

        grid $loc.frame.${key}_label -column 0 -row $idx -sticky nswe
        grid $loc.frame.$key -column 1 -row $idx -sticky nswe
    }

    # visualization tab
    ttk::frame $w.hlf.nb.vis
    $w.hlf.nb add $w.hlf.nb.vis -text "Visualization"
    set vis $w.hlf.nb.vis
    grid columnconfigure $vis 0 -weight 1
    grid rowconfigure $vis 0 -weight 1

    ttk::frame $vis.frame
    ttk::label $vis.frame.label -text "You must have at least a molecule and a map loaded." -justify center -anchor center
    pack $vis.frame.label -expand 1 -fill both -padx 4 -pady 4
    grid $vis.frame -column 0 -row 0 -sticky nswe
    grid columnconfigure $vis.frame 0 -weight 1

    # ligand tab
    ttk::frame $w.hlf.nb.ligands
    $w.hlf.nb add $w.hlf.nb.ligands -text "Ligand"
    set ligands $w.hlf.nb.ligands
    grid columnconfigure $ligands 0 -weight 1
    grid rowconfigure $ligands 0 -weight 1

    ttk::frame $ligands.frame
    ttk::label $ligands.frame.label -text "You must have at least a molecule and a map loaded." -justify center -anchor center
    pack $ligands.frame.label -expand 1 -fill both -padx 4 -pady 4
    grid $ligands.frame -column 1 -row 1 -columnspan 1 -sticky nswe
    grid columnconfigure $ligands.frame.label 0 -weight 1

    # MC-SILCS tab
    ttk::frame $w.hlf.nb.mcsilcs
    $w.hlf.nb add $w.hlf.nb.mcsilcs -text "MC-SILCS"
    set mcsilcs $w.hlf.nb.mcsilcs

    ttk::frame $mcsilcs.frame
    ttk::label $mcsilcs.frame.label -text "You must have at least a molecule and a map loaded." -justify center -anchor center
    pack $mcsilcs.frame.label -expand 1 -fill both -padx 4 -pady 4
    grid $mcsilcs.frame -column 1 -row 1 -columnspan 1 -sticky nswe
    grid columnconfigure $mcsilcs.frame.label 0 -weight 1

    # about tab
    ttk::frame $w.hlf.nb.about
    $w.hlf.nb add $w.hlf.nb.about -text "About"
    set about $w.hlf.nb.about
    grid columnconfigure $about 0 -weight 1
    grid rowconfigure $about 0 -weight 1

    ttk::frame $about.frame
    text $about.frame.text
    $about.frame.text insert 1.0 "This plugin integrates VMD (http://www.ks.uiuc.edu/Research/vmd/) with FragMaps from SILCS simulation.

Documentation may be found at
http://silcsbio.com

In the simplest case,

1) Load a structure into VMD.
2) Start this plugin.
3) Click the \"Visualize FragMap\" button.

Contact info@silcsbio.com for support.

Created by SilcsBio, 2016"
    ttk::scrollbar $about.frame.scrollbar -command "$about.frame.text yview"
    $about.frame.text conf -yscrollcommand "$about.frame.scrollbar set"

    grid $about.frame -column 0 -row 0 -sticky nswe -padx 10 -pady 10
    grid $about.frame.text -column 0 -row 0 -sticky nswe
    grid $about.frame.scrollbar -column 1 -row 0 -sticky nswe
    grid columnconfigure $about.frame 0 -weight 1
    grid rowconfigure $about.frame 0 -weight 1

    # build / grid a separator
    ttk::separator $w.hlf.sep2 -orient horizontal
    grid $w.hlf.sep2 -column 0 -row 3 -sticky nswe -pady 2 -padx 2

    #
    # build a control button panel
    #
    ttk::frame $w.hlf.controlButtons
    ttk::button $w.hlf.controlButtons.visualize -text "Visualize FragMap" -command { ::silcs::show_fragmap }
    ttk::button $w.hlf.controlButtons.loadligands -text "Load Ligands" -command { ::silcs::ligand_dialog }
    ttk::button $w.hlf.controlButtons.loadmcsilcs -text "Load MC-SILCS" -command { ::silcs::mcsilcs_dialog }
    ttk::button $w.hlf.controlButtons.quit -text "Exit FragMap Tools" -command { ::silcs::quit }
    grid $w.hlf.controlButtons -column 0 -row 4 -sticky ns -pady "0 10"
    grid $w.hlf.controlButtons.visualize -column 0 -row 0 -sticky nswe
    grid $w.hlf.controlButtons.loadligands -column 1 -row 0 -sticky nswe
    grid $w.hlf.controlButtons.loadmcsilcs -column 2 -row 0 -sticky nswe
    grid $w.hlf.controlButtons.quit -column 3 -row 0 -sticky nswe
    grid columnconfigure $w.hlf.controlButtons {0 1 2 3} -uniform ct1
}

proc ::silcs::init {} {
    # initialize the gui
    global env

    set ::silcs::mapdir "2b_gen_maps"
    set ::silcs::prefix ""
    set n_loaded_molecules [molinfo num]
    if { $n_loaded_molecules > 0 } {
        set topmol [molinfo top]
        set topfilename [molinfo top get name]
        set topname [string map {.pdb ""} $topfilename]
        set ::silcs::prefix $topname
    }

    # fragmap locations
    set fragmaps_enabled $::silcs::fragmaps_enabled
    set fragmaps_available $::silcs::fragmaps_available
    set nelem [llength $fragmaps_enabled]
    for {set i 0} {$i < $nelem} {incr i} {
        set key [lindex $fragmaps_enabled $i]
        set fragmap [dict get $fragmaps_available $key]
        set tv ${key}_fragmap_filename
        set flag ${key}_fragmap_visible
        set cutoff ${key}_fragmap_cutoff
        if { $key eq "excl" } {
            set ::silcs::$tv [file join "{mapdir}" [format "{prefix}.%s.map" $key]]
        } else {
            set ::silcs::$tv [file join "{mapdir}" [format "{prefix}.%s.gfe.map" $key]]
        }
        set ::silcs::$flag [dict get $fragmap visible]
        set ::silcs::$cutoff [dict get $fragmap cutoff]
    }

    # default checkbox options
    set ::silcs::surf_visible 0
    set ::silcs::cartoon_visible 0

    # delete existing maps
    if {[info exists ::silcs::maphash]} {
        unset ::silcs::maphash
    }
    if {[info exists ::silcs::ligandhash]} {
        unset ::silcs::ligandhash
        unset ::silcs::ligandsloaded
    }
    if {[info exists ::silcs::mcligandhash]} {
        unset ::silcs::mcligandhash
        unset ::silcs::mcligandsloaded
    }

    set ::silcs::mcligandsloaded {}
    set ::silcs::ligandsloaded {}
}

proc ::silcs::quit {} {
    variable w
    destroy $w
}

proc ::silcs::show_fragmap {} {
    ::silcs::refresh_fragmap

    variable w
    set vis $w.hlf.nb.vis
    set topmol [molinfo top]

    # load fragmaps using default setting
    set fragmaps_enabled $::silcs::fragmaps_enabled
    set fragmaps_available $::silcs::fragmaps_available
    set nelem [llength $fragmaps_enabled]
    for {set i 0} {$i < $nelem} {incr i} {
        set key [lindex $fragmaps_enabled $i]
        set fragmap [dict get $fragmaps_available $key]
        set label [dict get $fragmap label]
        set flag ${key}_fragmap_visible
        set cutoff ${key}_fragmap_cutoff
        ::silcs::loadfragmap $key cutoff [set ::silcs::$cutoff] flag [set ::silcs::$flag]
    }

    # select visualization tab
    $w.hlf.nb select $w.hlf.nb.vis
    mol top $topmol
    display resetview
    scale by 0.5
}

proc ::silcs::refresh_fragmap {} {
    variable w
    set vis $w.hlf.nb.vis

    destroy $vis.frame

    ttk::frame $vis.frame
    grid $vis.frame -column 0 -row 0 -sticky nwse -padx 10 -pady 5
    grid columnconfigure $vis 0 -weight 1

    ttk::label $vis.frame.header_maptype -text "FragMap Type" -anchor center
    ttk::label $vis.frame.header_maplevel -text "GFE Level" -anchor center

    set fragmaps_enabled $::silcs::fragmaps_enabled
    set fragmaps_available $::silcs::fragmaps_available
    set nelem [llength $fragmaps_enabled]
    for {set i 0} {$i < $nelem} {incr i} {
        set key [lindex $fragmaps_enabled $i]
        set fragmap [dict get $fragmaps_available $key]
        set label [dict get $fragmap label]
        set flag ${key}_fragmap_visible
        set cutoff ${key}_fragmap_cutoff
        set row [expr $i + 1]

        # skip if fragmap type is excl
        if {$key == "excl"} { continue }

        ttk::checkbutton $vis.frame.${key}_visible -variable ::silcs::$flag -onvalue 1 -offvalue 0 -command "::silcs::loadfragmap $key"
        ttk::label $vis.frame.${key}_label -text $label -anchor w
        ttk::frame $vis.frame.${key}Frame
        ttk::button $vis.frame.${key}Frame.minus -text "-" -width 2 -command "::silcs::updatelevel $key lower"
        scale $vis.frame.${key}Frame.scale -orient horizontal \
            -showvalue false -from $::silcs::maplevel_lower -to $::silcs::maplevel_upper -resolution 0.1 -digit 2 \
            -bg #7f7f7f -activebackground #7f7f7f \
            -variable ::silcs::$cutoff \
            -command "::silcs::updatelevel $key"
        ttk::button $vis.frame.${key}Frame.plus -text "+" -width 2 -command "::silcs::updatelevel $key upper"
        ttk::label $vis.frame.${key}_cutoff_label -textvariable ::silcs::$cutoff -anchor e -justify right
        ttk::label $vis.frame.${key}_cutoff_label_unit -text "kcal/mol" -anchor w

        grid $vis.frame.${key}_visible -column 0 -row $row -sticky nwe
        grid $vis.frame.${key}_label -column 1 -row $row -sticky nwe -padx 10
        grid $vis.frame.${key}Frame -column 2 -row $row -sticky nwe -padx 10
        grid $vis.frame.${key}_cutoff_label -column 3 -row $row -sticky nwe -padx 10
        grid $vis.frame.${key}_cutoff_label_unit -column 4 -row $row -sticky nwe

        grid $vis.frame.${key}Frame.minus -column 0 -row 0 -sticky nwse
        grid $vis.frame.${key}Frame.scale -column 1 -row 0 -sticky we
        grid $vis.frame.${key}Frame.plus -column 2 -row 0 -sticky nwse
        grid columnconfigure $vis.frame.${key}Frame 1 -weight 1
    }
    grid $vis.frame.header_maptype -column 1 -row 0 -sticky nswe -padx 10
    grid $vis.frame.header_maplevel -column 2 -row 0 -columnspan 1 -sticky nswe
    grid columnconfigure $vis.frame 4 -weight 1
    grid rowconfigure $vis.frame $row -weight 1

    incr row
    ttk::separator $vis.frame.sep -orient horizontal
    grid $vis.frame.sep -column 0 -row $row -columnspan 5 -sticky nswe -pady 4

    # display protein surface map options
    ttk::label $vis.frame.header_surftype -text "Surface Type" -anchor center
    ttk::label $vis.frame.header_inputmol -text "Molecule" -anchor center

    # fetch possible protein molecule ID
    # this list should not include molecules read by the plugin (map, ligands)
    set mollist [::silcs::mollist]
    set inputmol {}
    foreach mol $mollist {
        lappend inputmol "$mol: [molinfo $mol get name]"
    }

    ttk::checkbutton $vis.frame.surf_visible -variable ::silcs::surf_visible -onvalue 1 -offvalue 0 -command "::silcs::loadsurface surf"
    ttk::label $vis.frame.surf_label -text "Protein Surface"
    ttk::combobox $vis.frame.surf_mollist -values $inputmol -state readonly
    if {[llength $inputmol] && [$vis.frame.surf_mollist current] < 0} { $vis.frame.surf_mollist current 0 }

    ttk::checkbutton $vis.frame.cartoon_visible -variable ::silcs::cartoon_visible -onvalue 1 -offvalue 0 -command "::silcs::loadsurface cartoon"
    ttk::label $vis.frame.cartoon_label -text "Protein Cartoon"
    ttk::combobox $vis.frame.cartoon_mollist -values $inputmol -state readonly
    if {[llength $inputmol] && [$vis.frame.cartoon_mollist current] < 0} { $vis.frame.cartoon_mollist current 0 }

    ttk::checkbutton $vis.frame.excl_visible -variable ::silcs::excl_fragmap_visible -onvalue 1 -offvalue 0 -command "::silcs::loadfragmap excl"
    ttk::label $vis.frame.excl_label -text "Exclusion Map" -anchor w

    incr row
    grid $vis.frame.header_surftype -row $row -column 1 -sticky nswe
    grid $vis.frame.header_inputmol -row $row -column 2 -sticky nswe
    incr row
    grid $vis.frame.surf_visible -row $row -column 0 -sticky nswe
    grid $vis.frame.surf_label -row $row -column 1 -sticky nswe
    grid $vis.frame.surf_mollist -row $row -column 2 -sticky nswe
    incr row
    grid $vis.frame.cartoon_visible -row $row -column 0 -sticky nswe
    grid $vis.frame.cartoon_label -row $row -column 1 -sticky nswe
    grid $vis.frame.cartoon_mollist -row $row -column 2 -sticky nswe
    incr row
    grid $vis.frame.excl_visible -row $row -column 0 -sticky nswe
    grid $vis.frame.excl_label -row $row -column 1 -sticky nswe
}

proc ::silcs::loadfragmap {maptype args} {
    set fragmaps_available $::silcs::fragmaps_available
    set fragmap [dict get $fragmaps_available $maptype]

    # default argument
    set flag [set ::silcs::${maptype}_fragmap_visible]
    set cutoff [set ::silcs::${maptype}_fragmap_cutoff]
    if {$maptype eq "excl"} {
        set mesh 0
    } else {
        set mesh 1
    }

    array set opt [concat "cutoff $cutoff flag $flag mesh $mesh" $args]
    set cutoff $opt(cutoff)
    set flag $opt(flag)
    set mesh $opt(mesh)
    set mapdir $::silcs::mapdir
    set prefix $::silcs::prefix
    set mapformat [set ::silcs::${maptype}_fragmap_filename]
    set mapfile [string map "{{mapdir}} $mapdir {{prefix}} $prefix" $mapformat]

    if {![file exists $mapfile]} {
        puts "[dict get $fragmap label] ($mapfile) does not exists"
        return
    }

    # load fragmap if not already loaded
    array set maphash [array get ::silcs::maphash]
    if {![info exists maphash($maptype)]} {
        mol load map $mapfile
        mol rename top [dict get $fragmap label]
        set topmol [molinfo top]
        set ::silcs::maphash($maptype) [molinfo top]
        puts "loading fragmap: $maptype"

        mol modstyle 0 $topmol Isosurface ${cutoff} 0 0 ${mesh} 1 1
        mol modcolor 0 $topmol ColorID [dict get $fragmap color]
    } else {
        set topmol $::silcs::maphash($maptype)
    }
    if { $flag } {
        mol on $topmol
    } else {
        mol off $topmol
    }
}

proc ::silcs::loadsurface {surftype args} {
    # default argument
    variable w
    set vis $w.hlf.nb.vis
    set flag [set ::silcs::${surftype}_visible]
    set topmol [lindex [split [$vis.frame.${surftype}_mollist get] ":"] 0]

    array set opt [concat "flag $flag" $args]
    set flag $opt(flag)
    set ::silcs::${surftype}_visible $flag

    if {![info exists ::silcs::${surftype}_rep]} {
        if {$surftype == "surf"} {
            mol selection "( protein or nucleic ) and noh"
            mol representation Surf
            mol material Transparent
            mol color colorID 8
        }
        if {$surftype == "cartoon"} {
            mol selection "protein or nucleic"
            mol representation NewCartoon
            mol material Opaque
            mol color Name
        }
        mol addrep $topmol
        set ::silcs::${surftype}_rep [expr [lindex [molinfo $topmol get numreps] end] -1]
    }
    if { $flag } {
        mol showrep $topmol [set ::silcs::${surftype}_rep] on
    } else {
        mol showrep $topmol [set ::silcs::${surftype}_rep] off
    }
}

proc ::silcs::updatelevel {maptype level} {
    array set maphash [array get ::silcs::maphash]
    set cutoffvar ::silcs::${maptype}_fragmap_cutoff
    set currentlevel $$cutoffvar
    set offset 0.1
    if { $level == "lower" } {
        set level [expr $currentlevel - $offset]
    }
    if { $level == "upper" } {
        set level [expr $currentlevel + $offset]
    }
    if { $level < $::silcs::maplevel_lower } {
        set level $::silcs::maplevel_lower
    }
    if { $level > $::silcs::maplevel_upper } {
        set level $::silcs::maplevel_upper
    }

    if {[info exists maphash($maptype)]} {
        set topmol $::silcs::maphash($maptype)
        set mesh 1
        if {$maptype eq "excl"} {
            set mesh 0
        }
        mol modstyle 0 $topmol Isosurface ${level} 0 0 ${mesh} 1 1
        set $cutoffvar [expr [format "%.1f" $level]]
    }
}

proc ::silcs::ligand_dialog {} {
    # Show file dialog for loading more ligands molecules

    variable w
    set types {
        {{PDB Files}    {.pdb}}
        {{Mol2 Files}   {.mol2}}
        {{SDF Files}    {.sdf}}
    }
    set molfiles [tk_getOpenFile -filetypes $types \
        -multiple true -title "Load Ligands"]
    for {set i 0} {$i < [llength $molfiles]} {incr i} {
        set molfile [lindex $molfiles $i]
        set idx [expr $i + [llength $::silcs::ligandsloaded]]
        set ::silcs::mol_${idx}_visible 0
        if {$idx == 0} {
            set ::silcs::mol_${idx}_visible 1
        }
        if {![info exists ::silcs::ligandhash($molfile)]} {
            set ::silcs::ligandhash($molfile) {}
            lappend ::silcs::ligandsloaded $molfile
            ::silcs::loadligand $molfile
        }
    }

    if {[llength $molfiles] > 0} {
        ::silcs::refresh_ligands

        # select ligands tab
        $w.hlf.nb select $w.hlf.nb.ligands
    }
}

proc ::silcs::show_ligands {} {
    # Load ligands that matches with pattern provided by "liganddir"
    # variable and show the "Ligands" tab in the plugin window.

    variable w
    set ligands $w.hlf.nb.ligands
    set topmol [molinfo top]

    # list of ligands in the folder
    set molfiles [glob $::silcs::liganddir]
    set nmolfiles [llength $molfiles]
    for {set i 0} {$i < [llength $molfiles]} {incr i} {
        set molfile [lindex $molfiles $i]
        set ::silcs::mol_${i}_visible 0
        if {$i == 0} {
            set ::silcs::mol_${i}_visible 1
        }
        if {![info exists ::silcs::ligandhash($molfile)]} {
            set ::silcs::ligandhash($molfile) {}
            lappend ::silcs::ligandsloaded $molfile
            ::silcs::loadligand $molfile
        }
    }

    ::silcs::refresh_ligands

    # select ligands tab
    $w.hlf.nb select $w.hlf.nb.ligands
}

proc ::silcs::refresh_ligands {} {
    variable w
    set ligands $w.hlf.nb.ligands

    destroy $ligands.frame

    ttk::frame $ligands.frame
    grid $ligands.frame -column 0 -row 0 -sticky nwes -padx 10 -pady 10
    grid columnconfigure $ligands 0 -weight 1

    ttk::label $ligands.frame.header_ligandfn -text "Ligand Filename" -anchor center
    ttk::label $ligands.frame.header_lgfe -text "LGFE score" -anchor center
    ttk::label $ligands.frame.header_le -text "LE score" -anchor center
    ttk::label $ligands.frame.header_gfe_label -text "GFE" -anchor center

    set molfiles $::silcs::ligandsloaded
    set nelem [llength $molfiles]
    for {set i 0} {$i < $nelem} {incr i} {
        set key [lindex $molfiles $i]
        set molinfo $::silcs::ligandhash($key)
        set row [expr $i+1]
        set lgfe [format "%6.1f kcal/mol" [dict get $molinfo lgfe]]
        set le [format "%6.2f kcal/mol" [dict get $molinfo le]]
        set filename $key
        set labelcaplength 50
        if {[string length $filename] > $labelcaplength} {
            set end [string length $filename]
            set begin [expr $end - $labelcaplength]
            set filename "... [string range $filename $begin $end]"
        }

        ttk::checkbutton $ligands.frame.mol_${i}_visible -onvalue 1 -offvalue 0 -variable ::silcs::mol_${i}_visible -command "::silcs::toggleligand $key"
        ttk::label $ligands.frame.mol_${i}_label -text $filename -anchor w
        ttk::label $ligands.frame.mol_${i}_lgfe -text $lgfe -anchor w
        ttk::label $ligands.frame.mol_${i}_le -text $le -anchor w

        ttk::frame $ligands.frame.mol_${i}_gfe_frame
        set gfe_frame $ligands.frame.mol_${i}_gfe_frame
        ttk::checkbutton $gfe_frame.togglegfe_label -onvalue 1 -offvalue 0 -variable ::silcs::mol_${i}_gfe_label -command "::silcs::togglegfe $key" -text "Label"
        ttk::checkbutton $gfe_frame.togglegfe_color -onvalue 1 -offvalue 0 -variable ::silcs::mol_${i}_gfe_color -command "::silcs::togglegfe $key" -text "Color"

        ttk::frame $ligands.frame.mol_${i}_frame
        ttk::button $ligands.frame.mol_${i}_frame.zoom -text "Zoom" -command "::silcs::zoomligand $key"

        grid $ligands.frame.mol_${i}_visible -column 0 -row $row -sticky nwe
        grid $ligands.frame.mol_${i}_label -column 1 -row $row -sticky nwe
        grid $ligands.frame.mol_${i}_lgfe -column 2 -row $row -sticky nwe -padx 5
        grid $ligands.frame.mol_${i}_le -column 3 -row $row -sticky nwe -padx 5

        grid $gfe_frame -column 4 -row $row -sticky nswe -padx 5
        grid $gfe_frame.togglegfe_label -column 0 -row $row -sticky nwe
        grid $gfe_frame.togglegfe_color -column 1 -row $row -sticky nwe

        grid $ligands.frame.mol_${i}_frame -column 5 -row $row -sticky nwe -padx 5
        grid $ligands.frame.mol_${i}_frame.zoom -column 0 -row $row -sticky nwe
    }
    grid $ligands.frame.header_ligandfn -column 1 -row 0 -sticky nswe
    grid $ligands.frame.header_lgfe -column 2 -row 0 -sticky nswe
    grid $ligands.frame.header_le -column 3 -row 0 -sticky nswe
    grid $ligands.frame.header_gfe_label -column 4 -row 0 -sticky nswe
    grid rowconfigure $ligands.frame $row -weight 1

    incr row
    ttk::separator $ligands.frame.sep -orient horizontal
    grid $ligands.frame.sep -column 0 -columnspan 6 -row $row -sticky nswe -pady 4

    # pocket

    # fetch possible protein molecule ID
    # this list should not include molecules read by the plugin (map, ligands)
    set mollist [::silcs::mollist]
    set inputmol {}
    foreach mol $mollist {
        lappend inputmol "$mol: [molinfo $mol get name]"
    }

    incr row
    ttk::frame $ligands.frame.pocketframe
    set pocketframe $ligands.frame.pocketframe
    ttk::label $pocketframe.pocket_label -text "Show Protein Atoms Near the Selected Ligands:" -anchor center
    ttk::label $pocketframe.mol_label -text "Protein Molecule:"
    ttk::combobox $pocketframe.mollist -values $inputmol -state readonly
    $pocketframe.mollist current 0
    ttk::label $pocketframe.radius_label -text "Radius:" -anchor w
    ttk::entry $pocketframe.radius -width 4
    $pocketframe.radius insert 0 "5"
    ttk::label $pocketframe.radius_unit_label -text "Å"
    ttk::button $pocketframe.definepocket -text "Define Pocket" -command "::silcs::definepocket"

    grid $ligands.frame.pocketframe -column 0 -columnspan 6 -row $row -pady 5
    grid $pocketframe.pocket_label -column 0 -row 0 -columnspan 6 -sticky nswe
    grid $pocketframe.mol_label -column 0 -row 1 -sticky nswe
    grid $pocketframe.mollist -column 1 -row 1 -sticky nswe
    grid $pocketframe.radius_label -column 2 -row 1 -sticky nswe
    grid $pocketframe.radius -column 3 -row 1 -sticky nswe
    grid $pocketframe.radius_unit_label -column 4 -row 1 -sticky nswe
    grid $pocketframe.definepocket -column 5 -row 1 -sticky nswe

    grid columnconfigure $ligands.frame 5 -weight 2
    grid columnconfigure $ligands.frame 1 -weight 1
}

proc ::silcs::loadligand {molfile args} {
    #set cutoff [dict get $fragmap cutoff]
    #set flag [dict get $fragmap visible]
    set molfiles $::silcs::ligandsloaded
    set idx [lsearch $molfiles $molfile]

    if { $idx < 0 } {
        lappend ::silcs::ligandsloaded $molfile
        set idx [lsearch $molfiles $molfile]
        set ::silcs::mol_${idx}_visible 0
        set ::silcs::ligandhash($molfile) {}
    }

    if {![info exists ::silcs::mol_${idx}_gfe_label]} {
        set ::silcs::mol_${idx}_gfe_label 0
        set ::silcs::mol_${idx}_gfe_color 0
        set ::silcs::mol_${idx}_gfe_objects {}
    }
    set flag [set ::silcs::mol_${idx}_visible]

    array set opt [concat "flag $flag" $args]
    set flag $opt(flag)

    # load fragmap if not already loaded
    array set ligandhash [array get ::silcs::ligandhash]
    if {![dict exists $ligandhash($molfile) topmol]} {
        mol new $molfile
        mol modstyle 0 top Licorice 0.100000 10.000000 10.000000
        mol modselect 0 top "all and noh"
        set topmol [molinfo top]
        dict set ::silcs::ligandhash($molfile) topmol $topmol
        dict set ::silcs::ligandhash($molfile) visible 1

        # compute LGFE and LE
        set all [atomselect top "all"]
        set natoms [$all num]
        set lgfe 0
        set nclass 0
        set nheavy 0
        for {set i 0} {$i < $natoms} {incr i} {
          set sel [atomselect top "index $i"]
          set lgfe [expr $lgfe + [$sel get beta]]
          set sname [$sel get segname]
          set elem [string index [$sel get type] 0]
          if { $sname ne "NCLA" } {
            incr nclass
          }
          if { $elem ne "H" } {
            incr nheavy
          }
          $sel delete
        }
        $all delete

        # we stopped renormalizing LGFE scores in 2017
        # set factor [$nheavy / $nclass]
        set factor 1
        set lgfe [expr $lgfe * $factor]
        set le [expr $lgfe / $nclass]
        dict set ::silcs::ligandhash($molfile) lgfe $lgfe
        dict set ::silcs::ligandhash($molfile) le $le
        puts "loading ligand: $molfile"
    } else {
        set topmol [dict get $::silcs::ligandhash($molfile) topmol]
    }
    if { $flag } {
        mol on $topmol
        mol top $topmol
    } else {
        mol off $topmol
        mol top 0
    }
}

proc ::silcs::toggleligand {molfile args} {
    set molfiles $::silcs::ligandsloaded
    set idx [lsearch $molfiles $molfile]
    set flag [set ::silcs::mol_${idx}_visible]

    array set ligandhash [array get ::silcs::ligandhash]
    set topmol [dict get $::silcs::ligandhash($molfile) topmol]
    if { $flag } {
        mol on $topmol
    } else {
        mol off $topmol
    }
}

proc ::silcs::zoomligand {molfile args} {
    set molfiles $::silcs::ligandsloaded
    set idx [lsearch $molfiles $molfile]

    array set ligandhash [array get ::silcs::ligandhash]
    set topmol [dict get $::silcs::ligandhash($molfile) topmol]
    mol on $topmol
    mol top $topmol
    display resetview
    scale by 0.75
}

proc ::silcs::togglegfe {molfile args} {
    set molfiles $::silcs::ligandsloaded
    set idx [lsearch $molfiles $molfile]
    set labelflag [set ::silcs::mol_${idx}_gfe_label]
    set colorflag [set ::silcs::mol_${idx}_gfe_color]

    array set ligandhash [array get ::silcs::ligandhash]
    set topmol [dict get $::silcs::ligandhash($molfile) topmol]
    if { $labelflag } {
        set sel [atomselect $topmol "all"]
        set natoms [$sel num]
        $sel delete
        if {![molinfo $topmol get top]} {
            zoomligand $molfile
        }
        for {set i 0} {$i < $natoms} {incr i} {
            set sel [atomselect $topmol "index $i"]
            set b [$sel get beta]
            set sname [$sel get segname]
            if { $sname ne "NCLA" } {
                set bf [format "%6.1f" [expr {$b}]]
                if { $b <= 0.0 } {
                    set objectid [label_atom "index $i" $bf "green"]
                    lappend ::silcs::mol_${idx}_gfe_objects $objectid
                    #set objectid [sphere_atom "index $i" -$bf "green"]
                    #lappend ::silcs::mol_${idx}_gfe_objects $objectid
                }
                if {$b > 0.0 } {
                    set objectid [label_atom "index $i" $bf "red"]
                    lappend ::silcs::mol_${idx}_gfe_objects $objectid
                    #set objectid [sphere_atom "index $i" $bf "red"]
                    #lappend ::silcs::mol_${idx}_gfe_objects $objectid
                }
            }
            $sel delete
        }
    } else {
        set nobjects [llength [set ::silcs::mol_${idx}_gfe_objects]]
        for {set i 0} {$i < $nobjects} {incr i} {
            set objectid [lindex [set ::silcs::mol_${idx}_gfe_objects] $i]
            draw delete $objectid
        }
        set ::silcs::mol_${idx}_gfe_objects {}
    }

    if { $colorflag } {
        if {![molinfo $topmol get top]} {
            zoomligand $molfile
        }
        mol modcolor 0 $topmol Beta
        mol scaleminmax $topmol 0 $::silcs::maplevel_lower $::silcs::maplevel_upper
    } else {
        mol modcolor 0 $topmol Name
    }
}

proc ::silcs::definepocket {} {
    # default argument
    variable w
    set ligands $w.hlf.nb.ligands
    set radius [$ligands.frame.pocketframe.radius get]
    set protmol [lindex [split [$ligands.frame.pocketframe.mollist get] ":"] 0]

    set molfiles [array names ::silcs::ligandhash]
    set nelem [llength $molfiles]
    set atoms ""
    for {set i 0} {$i < $nelem} {incr i} {
        set key [lindex $molfiles $i]
        set molinfo $::silcs::ligandhash($key)
        set topmol [dict get $molinfo topmol]
        if {[set ::silcs::mol_${i}_visible]} {
            set sel [atomselect $topmol {not hydrogen}]
            set atoms [concat $atoms [atomsnearligand $protmol $sel $radius]]
            $sel delete
        }
    }
    set atoms [lsort -uniq $atoms]
    #puts $atoms
    if { $::silcs::pocketrep == -1 } {
        mol selection "(same residue as index $atoms) and noh"
        mol representation Licorice 0.100000 10.000000 10.000000
        mol material Opaque
        mol color Name
        mol addrep $protmol
        set ::silcs::pocketrep [expr [lindex [molinfo $protmol get numreps] end] -1]
    } else {
        mol modselect $::silcs::pocketrep $protmol "(same residue as index $atoms) and noh"
        mol selupdate $::silcs::pocketrep $protmol on
    }


}

proc ::silcs::mcsilcs_dialog {} {
    # Show file dialog for loading more ligands molecules

    variable w
    set types {
        {{PDB Files}    {.pdb}}
        {{Mol2 Files}   {.mol2}}
        {{SDF Files}    {.sdf}}
    }
    set molfiles [tk_getOpenFile -filetypes $types \
        -multiple true -title "Load Ligands"]
    for {set i 0} {$i < [llength $molfiles]} {incr i} {
        set molfile [lindex $molfiles $i]
        set idx [expr $i + [llength $::silcs::mcligandsloaded]]
        set ::silcs::mcmol_${idx}_visible 0
        if {$idx == 0} {
            set ::silcs::mcmol_${idx}_visible 1
        }
        if {![info exists ::silcs::mcligandhash($molfile)]} {
            set ::silcs::mcligandhash($molfile) {}
            lappend ::silcs::mcligandsloaded $molfile
            ::silcs::loadmcligand $molfile
        }
    }

    if {[llength $molfiles] > 0} {
        ::silcs::refresh_mcsilcs

        # select ligands tab
        $w.hlf.nb select $w.hlf.nb.mcsilcs
    }
}

proc ::silcs::show_mcsilcs {} {
    variable w
    set mcsilcs $w.hlf.nb.mcsilcs
    set topmol [molinfo top]

    # list of ligands in the folder
    set mol2files [glob $::silcs::mcsilcsdir]
    set nmolfiles [llength $mol2files]
    for {set i 0} {$i < [llength $mol2files]} {incr i} {
        set molfile [lindex $mol2files $i]
        set ::silcs::mcmol_${i}_visible 0
        if {$i == 0} {
            set ::silcs::mcmol_${i}_visible 1
        }
        if {![info exists ::silcs::mcligandhash($molfile)]} {
            set ::silcs::mcligandhash($molfile) {}
            lappend ::silcs::mcligandsloaded $molfile
            ::silcs::loadmcligand $molfile
        }
    }

    ::silcs::refresh_mcsilcs

    # select mc-silcs tab
    $w.hlf.nb select $w.hlf.nb.mcsilcs
}

proc ::silcs::refresh_mcsilcs {} {
    variable w
    set mcsilcs $w.hlf.nb.mcsilcs

    destroy $mcsilcs.frame

    ttk::frame $mcsilcs.frame
    grid $mcsilcs.frame -column 0 -row 0 -sticky nw -padx 10 -pady 10

    ttk::label $mcsilcs.frame.header_ligandfn -text "Ligand Filename" -anchor center
    ttk::label $mcsilcs.frame.header_conf -text "Conformation" -anchor center
    ttk::label $mcsilcs.frame.header_lgfe -text "LGFE" -anchor center

    set molfiles $::silcs::mcligandsloaded
    set nelem [llength $molfiles]
    for {set i 0} {$i < $nelem} {incr i} {
        set key [lindex $molfiles $i]
        set molinfo $::silcs::mcligandhash($key)
        set row [expr $i+1]
        set numframes [dict get $::silcs::mcligandhash($key) numframes]
        set filename $key
        set labelcaplength 50
        if {[string length $filename] > $labelcaplength} {
            set end [string length $filename]
            set begin [expr $end - $labelcaplength]
            set filename "... [string range $filename $begin $end]"
        }

        ttk::checkbutton $mcsilcs.frame.mcmol_${i}_visible -onvalue 1 -offvalue 0 -variable ::silcs::mcmol_${i}_visible -command "::silcs::loadmcligand $key"
        ttk::label $mcsilcs.frame.mcmol_${i}_label -text $filename -anchor w

        ttk::frame $mcsilcs.frame.mcmol${i}Frame
        ttk::button $mcsilcs.frame.mcmol${i}Frame.minus -text "<" -width 2 -command "::silcs::loadmcligand $key frame prev"
        scale $mcsilcs.frame.mcmol${i}Frame.scale -orient horizontal \
            -showvalue false -from 0 -to [dict get $::silcs::mcligandhash($key) numframes] -resolution 1 \
            -bg #7f7f7f -activebackground #7f7f7f -length 100 \
            -variable ::silcs::mcmol_${i}_frame \
            -command "::silcs::loadmcligand $key frame"
        ttk::button $mcsilcs.frame.mcmol${i}Frame.plus -text ">" -width 2 -command "::silcs::loadmcligand $key frame next"

        ttk::label $mcsilcs.frame.mcmol_${i}_frame_label -textvariable ::silcs::mcmol_${i}_frame -anchor e -justify right -width 5
        ttk::label $mcsilcs.frame.mcmol_${i}_lgfe -textvariable ::silcs::mcmol_${i}_lgfe -anchor w
        ttk::button $mcsilcs.frame.mcmol_${i}_zoom -text "Zoom" -command "::silcs::zoommcligand $key"

        grid $mcsilcs.frame.mcmol_${i}_visible -column 0 -row $row -sticky nswe
        grid $mcsilcs.frame.mcmol_${i}_label -column 1 -row $row -sticky nswe -padx 5

        grid $mcsilcs.frame.mcmol${i}Frame -column 2 -row $row -sticky we -padx 5
        grid $mcsilcs.frame.mcmol_${i}_frame_label -column 3 -row $row -sticky nswe -padx 5
        grid $mcsilcs.frame.mcmol_${i}_lgfe -column 4 -row $row -sticky nswe
        grid $mcsilcs.frame.mcmol_${i}_zoom -column 5 -row $row -sticky nswe -padx 5

        grid $mcsilcs.frame.mcmol${i}Frame.minus -column 0 -row 0 -sticky nwse
        grid $mcsilcs.frame.mcmol${i}Frame.scale -column 1 -row 0 -sticky we
        grid $mcsilcs.frame.mcmol${i}Frame.plus -column 2 -row 0 -sticky nwse
        grid columnconfigure $mcsilcs.frame.mcmol${i}Frame 1 -weight 1
    }
    grid $mcsilcs.frame.header_ligandfn -column 1 -row 0 -sticky nwse
    grid $mcsilcs.frame.header_conf -column 2 -columnspan 2 -row 0 -sticky nwse
    grid $mcsilcs.frame.header_lgfe -column 4 -row 0 -sticky nwse
    grid columnconfigure $mcsilcs.frame 5 -weight 1
}

proc ::silcs::loadmcligand {molfile args} {
    # default argument
    variable w
    set mcsilcs $w.hlf.nb.mcsilcs

    #set cutoff [dict get $fragmap cutoff]
    #set flag [dict get $fragmap visible]
    set molfiles $::silcs::mcligandsloaded
    set idx [lsearch $molfiles $molfile]
    set flag [set ::silcs::mcmol_${idx}_visible]
    set frame 0

    array set opt [concat "flag $flag frame $frame" $args]
    set flag $opt(flag)
    set frame $opt(frame)
    set ::silcs::mcmol_${idx}_visible $flag

    # load fragmap if not already loaded
    array set mcligandhash [array get ::silcs::mcligandhash]
    if {![dict exists $mcligandhash($molfile) topmol]} {
        #mol new $molfile first 0 last -1 step 1 waitfor all
        pdbbfactor $molfile
        mol modstyle 0 top Licorice 0.100000 10.000000 10.000000
        mol modselect 0 top "all and noh"
        lassign [molinfo top get {id numframes}] topmol numframes
        if {$numframes > 0} {
            molinfo $topmol set frame 0
        }
        dict set ::silcs::mcligandhash($molfile) topmol $topmol
        dict set ::silcs::mcligandhash($molfile) visible 1
        dict set ::silcs::mcligandhash($molfile) numframes $numframes
        dict set ::silcs::mcligandhash($molfile) frame -1
        set lgfearr {}
        for {set i 0} {$i < $numframes} {incr i} {
            set sel [atomselect top all]
            $sel frame $i
            set lgfe [lsum [$sel get user]]
            lappend lgfearr $lgfe
            $sel delete
        }
        dict set ::silcs::mcligandhash($molfile) lgfe $lgfearr
        puts "loading mc-ligand: $molfile $numframes"
    } else {
        set topmol [dict get $::silcs::mcligandhash($molfile) topmol]
        set numframes [dict get $::silcs::mcligandhash($molfile) numframes]
    }
    if { $flag } {
        mol on $topmol
        mol top $topmol
    } else {
        mol off $topmol
    }

    set currentframe [dict get $::silcs::mcligandhash($molfile) frame]
    set totalframe [dict get $::silcs::mcligandhash($molfile) numframes]
    if { $frame == "prev" } {
        set frame [expr $currentframe - 1]
    }
    if { $frame == "next" } {
        set frame [expr $currentframe + 1]
    }
    if { $frame < 0 } {
        set frame 0
    }
    if { $frame >= $totalframe } {
        set frame [expr $totalframe - 1]
    }

    if { [dict get $silcs::mcligandhash($molfile) frame] != $frame } {
        mol top $topmol
        molinfo $topmol set frame $frame
        dict set ::silcs::mcligandhash($molfile) frame $frame
        set ::silcs::mcmol_${idx}_frame $frame
        set ::silcs::mcmol_${idx}_lgfe [format "%6.1f kcal/mol" [lindex [dict get $::silcs::mcligandhash($molfile) lgfe] $frame]]
        puts "$molfile frame: $frame"
    }
}

proc ::silcs::zoommcligand {molfile args} {
    set molfiles $::silcs::mcligandsloaded
    set idx [lsearch $molfiles $molfile]
    set topmol [dict get $::silcs::mcligandhash($molfile) topmol]
    mol on $topmol
    mol top $topmol
    display resetview
    scale by 0.75
}

# utility procedures

# build list of molecules excluding molecules read by the plugin (map, ligand)
proc ::silcs::mollist {} {
    set mollist {}
    set excllist {}
    foreach key [array names ::silcs::maphash] {
        lappend excllist $silcs::maphash($key)
    }
    foreach key [array names ::silcs::ligandhash] {
        lappend excllist [dict get $silcs::ligandhash($key) topmol]
    }
    foreach key [array names ::silcs::mcligandhash] {
        lappend excllist [dict get $silcs::mcligandhash($key) topmol]
    }
    foreach mol [molinfo list] {
        if {[lsearch $excllist $mol] == -1} {
            lappend mollist $mol
        }
    }
    return $mollist
}

# label atoms
proc label_atom {selection_string label_string color} {
    set sel [atomselect top $selection_string]
    if {[$sel num] != 1} {
        error "label_atom: '$selection_string' must select 1 atom"
    }
    # get the coordinates of the atom
    lassign [$sel get {x y z}] coord
    # and draw the text
    draw color $color
    set objectid [draw text $coord $label_string thickness 3]
    # release memory
    $sel delete
    return $objectid
}

proc sphere_atom {selection_string label_string color} {
    set sel [atomselect top $selection_string]
    if {[$sel num] != 1} {
        error "label_atom: '$selection_string' must select 1 atom"
    }
    # get the coordinates of the atom
    lassign [$sel get {x y z}] coord
    # and draw the text
    draw color $color
    set objectid [draw sphere $coord radius [expr $label_string] resolution 10]
    # release memory
    $sel delete
    return $objectid
}

# load multi-frame pdb file, storing B factors from each frame in user.
proc pdbbfactor { fname } {
  mol new $fname waitfor all
  set all [atomselect top all]
  set natoms [$all num]
  set nclass 0
  set nheavy 0
  for {set i 0} {$i < $natoms} {incr i} {
    set sel [atomselect top "index $i"]
    set sname [$sel get segname]
    set elem [string index [$sel get type] 0]
    if { $sname ne "NCLA" } {
      incr nclass
    }
    if { $elem ne "H" } {
      incr nheavy
    }
    $sel delete
  }
  # we stopped renormalizing LGFE scores in 2017
  #set factor [expr double($nheavy) / $nclass]
  set factor 1

  set frame 0
  set in [open $fname r]
  set beta {}
  while { [gets $in line] != -1 } {
    switch -- [string range $line 0 3] {
      REMA {
        if {[string range $line 7 10] == "LGFE"} {
          set lgfe [string range $line 14 22]
        }
      }
      ENDM -
      END {
        # When REMARK LGFE is found but no B-factor column is populated,
        # assign the LGFE value to one of the b-factor entry, so it can be
        # summed to correct LGFE later.
        #
        # This LGFE score is already weighted,
        # LGFE = LGFE(raw) * n_heavy / n_classified
        # so no further weighting should not be applied here.
        #
        if {[lsum $beta] == 0.0 && $lgfe != 0} {
          lset beta 1 $lgfe
        }
        $all frame $frame
        $all set user $beta
        set beta {}
        incr frame
      }
      ATOM -
      HETA {
        lappend beta [expr [string range $line 61 66] * $factor]
      }
    }
  }
  $all delete
}

# return the sum of list elements
proc lsum {l} {
    set total 0.0
    foreach nxt $l {
        set total [expr {$total + $nxt}]
    }
    return $total
}

# return atom idx within cutoff from the selected atoms
proc atomsnearligand {prot ligand cutoff} {
  set atoms {}
  foreach coords [$ligand get {x y z}] {
    set x [lindex $coords 0]
    set y [lindex $coords 1]
    set z [lindex $coords 2]
    set sel [atomselect $prot "sqr(x-$x)+sqr(y-$y)+sqr(z-$z) < sqr($cutoff)"]
    set atoms [concat $atoms [$sel get index]]
    $sel delete
  }
  return [lsort -unique $atoms]
}
